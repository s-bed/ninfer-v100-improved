// Model-owned FP8 QPN requests and upload/resume, without whole-model allocation.
#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "core/vmm.h"
#include "models/qwen3_5/load/bindings.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/fp8/fp8_format.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <tuple>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
namespace ar=artifact;
namespace q=models::qwen3_5;
namespace loading=q::loading;
using test::artifact_fixture::Fixture;
using test::artifact_fixture::Json;
using test::artifact_fixture::put_word;
using test::artifact_fixture::require;
using test::artifact_fixture::rejects;

// CPU tests need only metadata. Sparse payloads avoid reading or allocating the
// hundreds of MiB of real-shape matrices before any device is involved.
struct LayoutFixture {
    Fixture file;
    std::uint64_t bytes=0;
    LayoutFixture() {
        file.root["components"]={{"text",{{"config",Json::object()},{"resources",Json::object()}}}};
        file.root["objects"]=Json::array();
        file.root["bindings"]=Json::object();
        file.root["uses"]=Json::array();
    }
    void tensor(std::string_view id,int n,int k) {
        const auto geometry=weight_geometry(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::RowScale,
                                            std::array<std::uint64_t,2>{std::uint64_t(n),std::uint64_t(k)});
        const auto offset=(bytes+255)/256*256;
        file.root["objects"].push_back({{"id",id},{"kind","tensor"},{"shape",{n,k}},
            {"format","fp8_e4m3fn_row_bf16"},{"layout","row_scale_v1"},
            {"offset",offset},{"bytes",geometry.bytes}});
        bytes=offset+geometry.bytes;
    }
    void binding(std::string_view name,std::string_view object,int k,int first,int last) {
        file.root["bindings"][std::string(name)]={{"parts",Json::array({
            {{"object",object},{"range",{std::uint64_t(first)*k,std::uint64_t(last)*k}}}})}};
    }
    void write(std::span<const std::byte> payload={}) {
        file.root["files"]=Json::array({{{"path",nullptr},{"payload_bytes",bytes}}});
        const std::string json=file.root.dump();
        std::array<std::byte,32> header{};
        constexpr std::array<unsigned char,8> magic{'N','I','N','F','E','R',0,3};
        for (std::size_t i=0;i<magic.size();++i) header[i]=std::byte(magic[i]);
        put_word(header,8,json.size(),8);
        header[16]=std::byte{0x71};
        const auto payload_at=(32+json.size()+4095)/4096*4096;
        std::ofstream output(file.entry,std::ios::binary|std::ios::trunc);
        output.exceptions(std::ios::badbit|std::ios::failbit);
        output.write(reinterpret_cast<const char*>(header.data()),header.size());
        output.write(json.data(),std::streamsize(json.size()));
        if (!payload.empty()) {
            require(payload.size()==bytes,"materializer fixture payload size differs");
            output.seekp(std::streamoff(payload_at));
            output.write(reinterpret_cast<const char*>(payload.data()),std::streamsize(payload.size()));
        } else {
            output.seekp(std::streamoff(payload_at+bytes-1));
            output.put('\0');
        }
    }
};

void define_model(LayoutFixture& f) {
    f.tensor("gdn",16384,5120);
    f.binding("query","gdn",5120,0,2048);
    f.binding("key","gdn",5120,2048,4096);
    f.binding("value","gdn",5120,4096,10240);
    f.binding("z","gdn",5120,10240,16384);
    f.tensor("residual",5120,6144);
    f.binding("residual","residual",6144,0,5120);
    f.binding("embedding_alias","residual",6144,0,5120);
    f.binding("head_alias","residual",6144,0,5120);
    f.binding("partial_a","residual",6144,0,2560);
    f.binding("partial_b","residual",6144,2560,5120);
    f.tensor("attention",14336,5120);
    f.binding("aq","attention",5120,0,6144);
    f.binding("ak","attention",5120,6144,7168);
    f.binding("ag","attention",5120,7168,13312);
    f.binding("av","attention",5120,13312,14336);
    f.tensor("mlp",34816,5120);
    f.binding("gate","mlp",5120,0,17408);
    f.binding("up","mlp",5120,17408,34816);
    f.tensor("down",5120,17408);
    f.binding("down","down",17408,0,5120);
    // A vocabulary matrix stays native even if a caller requests it directly.
    f.tensor("vocabulary",248320,5120);
    f.write();
}

q::ModelWeights bind_model(loading::Bindings& b) {
    q::ModelWeights m;
    q::BlockWeights block;
    q::GdnWeights g;
    g.query=b.parameter("query",{2048,5120});
    g.key=b.parameter("key",{2048,5120});
    g.value=b.parameter("value",{6144,5120});
    g.z=b.parameter("z",{6144,5120});
    g.output=b.parameter("residual",{5120,6144});
    block.mixer=g;
    q::DenseWeights dense;
    dense.gate=b.parameter("gate",{17408,5120});
    dense.up=b.parameter("up",{17408,5120});
    dense.down=b.parameter("down",{5120,17408});
    block.ffn=dense;
    m.text.layers.push_back(block);
    q::BlockWeights attention_block;
    q::AttentionWeights a;
    a.query=b.parameter("aq",{6144,5120});
    a.key=b.parameter("ak",{1024,5120});
    a.gate=b.parameter("ag",{6144,5120});
    a.value=b.parameter("av",{1024,5120});
    attention_block.mixer=a;
    m.text.layers.push_back(attention_block);
    return m;
}

bool requested(const ar::MaterializationPlan& plan,ar::ObjectHandle object) {
    const auto found=std::find_if(plan.device_objects.begin(),plan.device_objects.end(),
        [&](const auto& placement){return placement.object==object;});
    require(found!=plan.device_objects.end(),"CPU request test lost its device object");
    return found->fp8_qpn;
}

void cpu_selection() {
    LayoutFixture f;
    define_model(f);
    ar::Reader reader(f.file.entry);
    constexpr std::array volta{ops::DeviceTier::Volta};
    constexpr std::array ampere{ops::DeviceTier::Ampere};
    const bool expect=ops::v100_ported_kernels_enabled() && ops::kBuildVolta;
    {
        ar::Binder binder(reader);
        loading::Bindings b(binder);
        const auto model=bind_model(b);
        loading::apply_device_layouts(b,model,volta);
        const auto plan=std::move(binder).finish();
        const auto resume_copy=plan;
        for (const auto* id: {"gdn","attention","residual","mlp","down"}) {
            require(requested(plan,reader.find(id))==expect,"a complete projection group lost its request");
            require(requested(resume_copy,reader.find(id))==expect,"copied resume placement lost its request");
        }
        require(plan.device_objects.size()==5,"layout annotation added a duplicate parent");
        require(reader.geometry(reader.find("gdn")).layout==QuantLayout::RowScale,
                "model annotation changed stored artifact geometry");
    }
    for (bool embedding : {false,true}) {
        ar::Binder binder(reader);
        loading::Bindings b(binder);
        auto model=bind_model(b);
        const auto alias=b.parameter(embedding?"embedding_alias":"head_alias",{5120,6144});
        if (embedding) model.text.token_embedding=alias;
        else model.text.output_head=alias;
        loading::apply_device_layouts(b,model,volta);
        const auto plan=std::move(binder).finish();
        require(!requested(plan,reader.find("residual")),"a tied embedding/head parent was prepacked");
        require(requested(plan,reader.find("gdn"))==expect,"a veto leaked into an unrelated parent");
    }
    for (bool embedding : {false,true}) {
        ar::Binder binder(reader);
        loading::Bindings b(binder);
        auto model=bind_model(b);
        const auto id=std::get<q::GdnWeights>(model.text.layers[0].mixer).output;
        if (embedding) model.text.token_embedding=id;
        else model.text.output_head=id;
        loading::apply_device_layouts(b,model,volta);
        require(!requested(std::move(binder).finish(),reader.find("residual")),
                "an embedding/head WeightId alias bypassed its consumer veto");
    }
    {
        ar::Binder binder(reader);
        loading::Bindings b(binder);
        const auto model=bind_model(b);
        loading::apply_device_layouts(b,model,ampere);
        const auto plan=std::move(binder).finish();
        for (const auto& placement:plan.device_objects)
            require(!placement.fp8_qpn,"a non-Volta rank requested FP8 packing");
    }
    {
        ar::Binder binder(reader);
        loading::Bindings b(binder);
        q::ModelWeights model;
        // The union covers all rows, but these are two independent residual
        // consumers. Neither may form a partial native_weight of layout7.
        for (const auto* name:{"partial_a","partial_b"}) {
            q::BlockWeights block;
            q::AttentionWeights a;
            a.output=b.parameter(name,{2560,6144});
            block.mixer=a;
            model.text.layers.push_back(block);
        }
        loading::apply_device_layouts(b,model,volta);
        require(!requested(std::move(binder).finish(),reader.find("residual")),
                "independent partial projection unions were treated as complete consumer groups");
    }
    {
        ar::Binder binder(reader);
        const auto source=reader.find("residual");
        const auto selected=binder.select_device(source,{.rows={{0,5120}},.columns={{0,6144}}},0);
        rejects([&]{binder.fp8_qpn_device(selected);},"a selected/TP replica accepted QPN packing");
        const auto plan=std::move(binder).finish();
        require(plan.device_objects.size()==1 && !plan.device_objects[0].fp8_qpn,
                "an unsupported selected request survived placement");
    }
    {
        ar::Binder binder(reader);
        binder.require_device(reader.find("vocabulary"));
        rejects([&]{binder.fp8_qpn_device(reader.find("vocabulary"));},
                "the unqualified full vocabulary shape accepted FP8 packing");
    }
}

void cpu_qpn_geometry_factory() {
    const auto values=[](const WeightGeometry& g) {
        return std::tie(g.format,g.shape,g.elements,g.bytes,g.alignment,g.padded_columns,
            g.group_size,g.code_bytes_per_row,g.high_bytes_per_row,g.scale_bytes_per_row,
            g.row_stride_bytes,g.code_bytes,g.high_offset,g.high_bytes,g.scale_offset,
            g.scale_bytes,g.divisor_offset,g.divisor_count);
    };
    const auto compare=[&](QType format,QuantLayout native_layout,
                           std::array<std::uint64_t,2> shape,std::uint64_t divisors=1) {
        const auto native=weight_geometry(format,native_layout,shape,divisors);
        const auto packed=weight_geometry(format,QuantLayout::VoltaQpnPrepacked,shape,divisors);
        require(packed.layout==QuantLayout::VoltaQpnPrepacked && values(native)==values(packed),
                "QPN geometry factory changed native format/shape/planes/scales/divisors");
    };
    compare(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::RowScale,{14336,5120});
    compare(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::RowScale,{34816,5120});
    compare(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::RowScale,{32,128});
    compare(QType::NVFP4,QuantLayout::BlockScaleK16M128x4,{256,64},2);
    compare(QType::NVFP4,QuantLayout::BlockScaleK16M128x4,{34816,5120},2);
    for (const auto shape:{std::array<std::uint64_t,2>{33,128},
                           std::array<std::uint64_t,2>{32,64},
                           std::array<std::uint64_t,2>{0,128}})
        rejects<std::invalid_argument>([&] {
            (void)weight_geometry(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::VoltaQpnPrepacked,shape);
        },"invalid FP8 QPN tile geometry was accepted");
    rejects<std::invalid_argument>([] {
        (void)weight_geometry(QType::BF16,QuantLayout::VoltaQpnPrepacked,
                              std::array<std::uint64_t,2>{32,128});
    },"unsupported QPN format was accepted");
    rejects<std::invalid_argument>([] {
        (void)weight_geometry(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::VoltaQpnPrepacked,
                              std::array<std::uint64_t,1>{128});
    },"a nonmatrix QPN factory shape was accepted");
    rejects<std::invalid_argument>([] {
        (void)weight_geometry(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::VoltaQpnPrepacked,
                              std::array<std::uint64_t,2>{32,128},2);
    },"FP8 QPN acquired NVFP4 divisors");
    for (std::uint64_t divisors:{0,3,4})
        rejects<std::invalid_argument>([&] {
            (void)weight_geometry(QType::NVFP4,QuantLayout::VoltaQpnPrepacked,
                                  std::array<std::uint64_t,2>{256,64},divisors);
        },"invalid packed NVFP4 divisor geometry was accepted");
    rejects<std::overflow_error>([] {
        (void)weight_geometry(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::VoltaQpnPrepacked,
            std::array<std::uint64_t,2>{std::numeric_limits<std::uint64_t>::max()-31,128});
    },"QPN factory lost checked element-count arithmetic");
}

void cpu_packed_metadata() {
    constexpr int n=256,k=128;
    auto geometry=weight_geometry(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::RowScale,
        std::array<std::uint64_t,2>{n,k});
    std::vector<std::byte> bytes(geometry.bytes);
    WeightParent parent{geometry,bytes.data()};
    const WeightView full{geometry.shape,{{&parent,0,geometry.elements}}};
    (void)ops::detail::validate_fp8_weight(native_weight(full),"native CPU metadata fixture");
    parent.geometry.layout=QuantLayout::VoltaQpnPrepacked;
    const auto planes=weight_row_planes({&parent,0,geometry.elements});
    const auto packed=native_weight(full);
    require(!planes.swizzled_scales && planes.scales==bytes.data()+geometry.scale_offset,
            "packed FP8 scales were treated as swizzled NVFP4 scales");
    require(packed.scale_dtype==DType::BF16 && packed.weight_divisors==nullptr &&
            packed.scale_ne[0]==n && packed.scale_nb[0]==2 && packed.scale_nb[1]==n*2,
            "packed FP8 native metadata acquired a different scale contract");
    for (int row:{0,1,31,32,n-1})
        require(weight_scale_offset(parent.geometry,row,0)==geometry.scale_offset+std::size_t(row)*2,
                "packed FP8 logical scale offset changed");
    rejects<std::invalid_argument>([&] {
        (void)native_weight(WeightView{{n/2,k},{{&parent,0,std::uint64_t(n/2)*k}}});
    },"a partial FP8 packed parent became a native projection");
    rejects<std::invalid_argument>([&] {
        (void)ops::detail::validate_fp8_weight(packed,"strict embedding/topk consumer");
    },"an unadapted strict FP8 consumer accepted packed bytes");
}

// Inverse of the code permutation, expressed per physical sixteen-byte slice.
std::size_t native_slice(std::size_t tuple,int k) {
    const int lane=int(tuple%32),slice=int((tuple/32)%8);
    const int block=int((tuple/256)%(k/128));
    const int tile=int(tuple/(std::size_t(k/128)*256));
    const int row=tile*32+((lane>>2)&3)*8+(lane&3)+(lane&16?4:0);
    return std::size_t(row)*k+block*128+slice*16;
}

int gpu_materialization() {
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || !count || !ops::current_device_volta()) return 77;
    DeviceContext device;
    if (!vmm::supported(device.rank(0).device)) return 77;
    constexpr int n=5120,k=6144;
    LayoutFixture f;
    f.tensor("projection",n,k);
    f.binding("projection","projection",k,0,n);
    const auto geometry=weight_geometry(QType::FP8_E4M3FN_ROW_BF16,QuantLayout::RowScale,
        std::array<std::uint64_t,2>{n,k});
    std::vector<std::byte> source(geometry.bytes);
    for (std::size_t i=0;i<geometry.code_bytes;++i) {
        const unsigned code=unsigned((i*17+i/127)%254);
        source[i]=std::byte(code+(code>=127));
    }
    for (int row=0;row<n;++row)
        put_word(source,geometry.scale_offset+std::size_t(row)*2,0x3580+(row%31)*19,2);
    f.write(source);
    auto owner=std::make_shared<ar::Reader>(f.file.entry);
    const auto& reader=*owner;
    ar::Binder binder(reader);
    const auto reference=binder.parameter("projection",{n,k});
    const auto object=reader.find("projection");
    binder.fp8_qpn_device(object);
    const auto host=binder.host_object(object);
    require(std::equal(host.begin(),host.end(),source.begin()),"native Host bytes changed before upload");
    auto backing=ar::materialize(reader,std::move(binder).finish(),device,nullptr,nullptr,
        ar::MaterializeOptions{.suspendable=true,.retained_reader=owner});
    const auto& parent=backing.device_parent(object);
    const bool packed=ops::v100_ported_kernels_enabled();
    require(parent.geometry.layout==(packed?QuantLayout::VoltaQpnPrepacked:QuantLayout::RowScale),
            "device parent label differs from its packing policy");
    const auto w=native_weight(ar::bind_view(reference,backing));
    require(w.scale_dtype==DType::BF16 && w.weight_divisors==nullptr,
            "FP8 packed parent acquired NVFP4 scale/divisor metadata");
    std::vector<std::byte> device_bytes(source.size());
    for (int upload=0;upload<2;++upload) {
        if (upload) {
            CUDA_CHECK(cudaMemset(const_cast<std::byte*>(parent.data),0xcc,parent.geometry.bytes));
            // cudaMemset may still be executing on the legacy stream. Quiesce
            // this intentional write before suspend unmaps its virtual address.
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
            const auto* fixed_address=parent.data;
            backing.release_device_backing();
            require(!backing.device_backed(),"suspend retained the weight device backing");
            backing.restore_device_backing(device);
            require(backing.device_backed() && parent.data==fixed_address,
                    "resume changed the prepared weight address");
            const auto stats=backing.upload_device_objects_again(device);
            require(stats.h2d_bytes>=geometry.bytes,"resume did not reload the complete native parent");
        }
        CUDA_CHECK(cudaMemcpy(device_bytes.data(),parent.data,device_bytes.size(),cudaMemcpyDeviceToHost));
        if (packed) {
            for (std::size_t tuple=0;tuple<geometry.code_bytes/16;++tuple)
                require(std::equal(device_bytes.begin()+tuple*16,device_bytes.begin()+tuple*16+16,
                                   source.begin()+native_slice(tuple,k)),
                        "materializer/resume permutation differs from native source bytes");
        } else require(device_bytes==source,"native policy transformed encoded FP8 bytes");
        require(std::equal(device_bytes.begin()+geometry.scale_offset,device_bytes.end(),
                           source.begin()+geometry.scale_offset),"packing changed logical BF16 scales");
        require(std::equal(backing.host_bytes(object).begin(),backing.host_bytes(object).end(),source.begin()),
                "device packing changed retained Host bytes");
        require(reader.read_object(object)==source,"device packing changed native artifact bytes");
    }
    return 0;
}
}

int main(int argc,char** argv) {
    const bool gpu=argc==3 && std::string_view(argv[2])=="--gpu";
    if ((argc!=2 && !gpu) || (std::string_view(argv[1])!="native" && std::string_view(argv[1])!="ported")) {
        std::cerr<<"usage: ninfer_v100_fp8_loading_test native|ported [--gpu]\n";
        return 2;
    }
    try {
        ops::set_v100_kernel_policy(std::string_view(argv[1])=="ported");
        cpu_selection();
        cpu_qpn_geometry_factory();
        cpu_packed_metadata();
        if (gpu) return gpu_materialization();
        std::cout<<"PASS FP8 model-owned complete-group selection and placement requests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"FP8 loading: "<<error.what()<<'\n';
        return 1;
    }
}
