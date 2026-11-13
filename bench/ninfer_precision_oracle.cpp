// Independent row-streamed represented-weight oracle. No production dequantizer
// or GPU reference is linked. Dot products retain FP64 across the entire K axis.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cblas.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
using json=nlohmann::json;
namespace fs=std::filesystem;
struct Mapping {
    int fd=-1;std::size_t bytes=0;const unsigned char* data=nullptr;
    explicit Mapping(const fs::path& path) {
        fd=open(path.c_str(),O_RDONLY);if(fd<0) throw std::runtime_error("cannot open "+path.string());
        struct stat status;if(fstat(fd,&status)) throw std::runtime_error("stat failed");
        bytes=status.st_size;
        data=static_cast<const unsigned char*>(mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0));
        if(data==MAP_FAILED) throw std::runtime_error("mmap failed");
    }
    ~Mapping(){if(data && data!=MAP_FAILED) munmap(const_cast<unsigned char*>(data),bytes);if(fd>=0)close(fd);}
    unsigned char at(std::size_t i)const {if(i>=bytes)throw std::runtime_error("oracle code range invalid");return data[i];}
    std::uint16_t u16(std::size_t i)const {return at(i)|(unsigned(at(i+1))<<8);}
    float f32(std::size_t i)const {std::uint32_t b=at(i)|(unsigned(at(i+1))<<8)|(unsigned(at(i+2))<<16)|(unsigned(at(i+3))<<24);return std::bit_cast<float>(b);}
};
double bf16(std::uint16_t value){return std::bit_cast<float>(std::uint32_t(value)<<16);}
double e4m3(unsigned byte) {
    unsigned magnitude=byte&127,exp=magnitude>>3,mantissa=magnitude&7;
    if(magnitude==127) return std::numeric_limits<double>::quiet_NaN();
    double value=exp?std::ldexp(1.0+mantissa/8.0,int(exp)-7):std::ldexp(double(mantissa),-9);
    return byte&128?-value:value;
}
struct Weight {
    json meta;Mapping codes,scales;int n,k,q,layout,divisor_rows;std::vector<double> divisors;
    explicit Weight(const fs::path& dir):meta(json::parse(std::ifstream(dir/"weight.json"))),codes(dir/"codes.bin"),scales(dir/"scales.bin") {
        n=meta["n"];k=meta["k"];q=meta["qtype"];layout=meta["layout"];
        divisor_rows=meta["weight_divisor_rows"];divisors=meta["weight_divisors"].get<std::vector<double>>();
        if(q!=7 && q!=8) throw std::runtime_error("unsupported oracle format");
        if(layout!=2 && layout!=3 && layout!=4 && layout!=7) throw std::runtime_error("unsupported oracle layout");
    }
    // Native nibble order, scale-block geometry and the two QPN permutations
    // are decoded directly from the storage definition, independently of kernels.
    float at(int row,int col)const {
        bool packed=layout==4 || layout==7;
        int lane=((row%32)/8)*4+(row&3)+((row&4)?16:0);
        if(q==7) {
            unsigned nibble,scale;
            if(packed) {
                constexpr int order[16]={0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15};
                int position=0;while(order[position]!=(col&15))position++;
                std::size_t group=(std::size_t(row/32)*(k/16)+col/16)*32+lane;
                auto byte=codes.at(group*8+position/2);
                nibble=(byte>>(4*(position&1)))&15;scale=scales.at(group);
            } else {
                auto byte=codes.at(std::size_t(row)*(k/2)+col/2);
                nibble=(byte>>(4*(col&1)))&15;
                std::size_t offset=(std::size_t(row/128)*(k/64)+col/64)*512+
                    (row&31)*16+((row&127)/32)*4+((col/16)&3);
                scale=scales.at(offset);
            }
            constexpr double levels[8]={0,.5,1,1.5,2,3,4,6};
            double value=levels[nibble&7]*(nibble&8?-1:1)*e4m3(scale)/divisors.at(row/divisor_rows);
            // Contract materializes the represented quantized weight in FP32.
            return float(value);
        }
        std::size_t offset=std::size_t(row)*k+col;
        if(packed) offset=(((std::size_t(row/32)*(k/128)+col/128)*8+(col%128)/16)*32+lane)*16+col%16;
        return float(e4m3(codes.at(offset))*bf16(scales.u16(std::size_t(row)*2)));
    }
};
struct Metric {
    long double squared=0,reference_squared=0,bias=0;
    double maximum=0,reference_max=0;std::uint64_t count=0,gross=0,nonfinite=0;
    int worst_row=0,worst_column=0;double worst_reference=0,worst_actual=0;
    void add(double expected,double actual,int row,int col) {
        ++count;
        if(!std::isfinite(expected)||!std::isfinite(actual)){++nonfinite;return;}
        double error=actual-expected;
        squared+=error*error;reference_squared+=expected*expected;bias+=error;
        reference_max=std::max(reference_max,std::abs(expected));
        if(std::abs(error)>maximum){maximum=std::abs(error);worst_row=row;worst_column=col;worst_reference=expected;worst_actual=actual;}
        if(std::abs(error)>std::max(.01,.05*std::abs(expected))) ++gross;
    }
    json result()const {
        return {{"count",count},{"relative_l2",double(std::sqrt(squared/std::max(reference_squared,1e-300L)))},
            {"rmse",double(std::sqrt(squared/std::max<std::uint64_t>(count,1)))},
            {"max_abs",maximum},{"reference_max_abs",reference_max},{"signed_bias",double(bias/std::max<std::uint64_t>(count,1))},
            {"gross_errors",gross},{"gross_definition","abs(error) > max(0.01, 0.05*abs(reference)); descriptive, not a replacement tolerance"},
            {"nonfinite",nonfinite},{"worst",{{"row",worst_row},{"column",worst_column},{"reference",worst_reference},{"actual",worst_actual}}}};
    }
};
void evaluate(const fs::path& canonical,const fs::path& candidate,const fs::path& output) {
    Weight reference(canonical),other(candidate);
    int n=reference.meta["output_rows"],k=reference.k;
    if(other.n!=reference.n || other.k!=k || other.meta["family"]!=reference.meta["family"]) throw std::runtime_error("unpaired operator");
    fs::path fixture=reference.meta["fixture"];
    Mapping activations(fixture/"activation.bf16");
    const int columns=fs::exists(candidate/"width-2048.bin")?2048:fs::exists(candidate/"width-1024.bin")?1024:128;
    std::vector<double> x(std::size_t(k)*columns);
    for(std::size_t i=0;i<x.size();++i)x[i]=bf16(activations.u16((i*2)%activations.bytes));
    std::vector<int> widths{1,2,4,6,8,16,32,33,63,64,65,128};
    for(int extra:{1024,2048})if(fs::exists(candidate/("width-"+std::to_string(extra)+".bin")))widths.push_back(extra);
    std::vector<std::unique_ptr<Mapping>> outputs;
    for(int width:widths) outputs.emplace_back(std::make_unique<Mapping>(candidate/("width-"+std::to_string(width)+".bin")));
    int element=other.meta["output_element_bytes"];
    std::unique_ptr<Mapping> residual;
    std::string family=reference.meta["family"];
    if(family=="linear_add") residual=std::make_unique<Mapping>(fixture/"residual.bin");
    std::vector<Metric> metrics(widths.size());
    double weight_max=0;std::uint64_t different_weights=0;
    constexpr int tile_rows=32;
    std::vector<double> sums(tile_rows*columns),up(tile_rows*columns);
    std::vector<double> row_weights(tile_rows*k),up_weights(tile_rows*k);
    double naive_max_abs=0,naive_max_bound_ratio=0;std::uint64_t naive_cells=0;
    openblas_set_num_threads(2);
    for(int base=0;base<n;base+=tile_rows) {
        int rows=std::min(tile_rows,n-base);
        for(int local=0;local<rows;++local) for(int j=0;j<k;++j) {
            int row=base+local;
            float a=reference.at(row,j),b=other.at(row,j);
            if(std::bit_cast<std::uint32_t>(a)!=std::bit_cast<std::uint32_t>(b)) ++different_weights;
            weight_max=std::max(weight_max,std::abs(double(a)-b));
            row_weights[local*k+j]=a;
            if(family=="linear_swiglu") {
                a=reference.at(row+n,j);b=other.at(row+n,j);
                if(std::bit_cast<std::uint32_t>(a)!=std::bit_cast<std::uint32_t>(b))++different_weights;
                weight_max=std::max(weight_max,std::abs(double(a)-b));up_weights[local*k+j]=a;
            }
        }
        // Independent FP64 GEMM, complete K, only a 32-row weight tile resident.
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,columns,k,1,
                    row_weights.data(),k,x.data(),k,0,sums.data(),columns);
        if(family=="linear_swiglu")cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,columns,k,1,
                    up_weights.data(),k,x.data(),k,0,up.data(),columns);
        for(int local=0;local<rows;++local) for(int col=0;col<columns;++col) {
            int row=base+local;double expected=sums[local*columns+col];
            double upper=up[local*columns+col];
            bool scalar_row=row==0 || row==n-1 || (std::int64_t(row)*31)/(n-1)!=(std::int64_t(row-1)*31)/(n-1);
            if(scalar_row) {
                // Direct scalar FP64 oracle on 32 fixed rows, every column,
                // complete K. Cross-check the accelerated full-matrix oracle
                // against a floating-point summation bound, independently of
                // the much wider production Op tolerance.
                double scalar=0,scalar_up=0,absolute_sum=0,absolute_up=0;
                for(int j=0;j<k;++j) {
                    double product=row_weights[local*k+j]*x[std::size_t(col)*k+j];
                    scalar+=product;absolute_sum+=std::abs(product);
                    if(family=="linear_swiglu") {
                        product=up_weights[local*k+j]*x[std::size_t(col)*k+j];
                        scalar_up+=product;absolute_up+=std::abs(product);
                    }
                }
                double gamma=(k*std::numeric_limits<double>::epsilon())/(1-k*std::numeric_limits<double>::epsilon());
                auto check=[&](double fast,double slow,double absolute) {
                    double error=std::abs(fast-slow),bound=2*gamma*absolute+1e-300;
                    naive_max_abs=std::max(naive_max_abs,error);
                    naive_max_bound_ratio=std::max(naive_max_bound_ratio,error/bound);
                    if(error>bound)throw std::runtime_error("FP64 BLAS disagrees with independent scalar oracle beyond its summation error bound");
                };
                check(expected,scalar,absolute_sum);expected=scalar;++naive_cells;
                if(family=="linear_swiglu"){check(upper,scalar_up,absolute_up);upper=scalar_up;}
            }
            if(family=="linear_swiglu")expected=(expected/(1+std::exp(-expected)))*upper;
            if(residual) {
                auto offset=((std::size_t(col)*n+row)*element)%residual->bytes;
                expected+=element==2?bf16(residual->u16(offset)):residual->f32(offset);
            }
            for(std::size_t i=0;i<widths.size();++i) if(col<widths[i]) {
                auto offset=(std::size_t(col)*n+row)*element;
                double actual=element==2?bf16(outputs[i]->u16(offset)):outputs[i]->f32(offset);
                metrics[i].add(expected,actual,row,col);
            }
        }
    }
    std::ofstream out(output,std::ios::app);
    for(std::size_t i=0;i<widths.size();++i) {
        auto result=metrics[i].result();
        result["operator"]=candidate.filename().string();result["engine"]=candidate.parent_path().parent_path().parent_path().filename().string();
        result["width"]=widths[i];result["family"]=family;result["qtype"]=reference.q;
        result["coverage"]="all output rows and columns; complete K; FP64 BLAS with 32-row represented-weight tiles";
        result["represented_weight_differences"]=different_weights;result["represented_weight_max_abs_difference"]=weight_max;
        result["scalar_fp64_cells"]=naive_cells;result["blas_vs_scalar_max_abs"]=naive_max_abs;result["blas_vs_scalar_max_summation_bound_ratio"]=naive_max_bound_ratio;
        result["oracle"]="independent codec, FP32 represented weights, BF16 represented input, complete FP64 GEMM; naive scalar FP64 on 32 fixed rows/all columns, cross-checked against summation bounds; FP64 SiLU/product or residual addition";
        out<<result.dump()<<'\n'<<std::flush;
    }
}
std::pair<unsigned,double> maximum_token(const Mapping& file,std::uint64_t offset,unsigned vocab) {
    double best=-std::numeric_limits<double>::infinity(),second=best;unsigned id=0;
    for(unsigned i=0;i<vocab;++i) {
        double value=bf16(file.u16(offset+std::uint64_t(i)*2));
        if(!std::isfinite(value))throw std::runtime_error("non-finite observed vocabulary row");
        if(value>best){second=best;best=value;id=i;}else if(value>second)second=value;
    }
    return {id,best-second};
}
void distributions(const json& jobs,const fs::path& output) {
    std::ofstream out(output,std::ios::app);
    for(auto item:jobs) {
        Mapping a(item["left_file"].get<std::string>()),b(item["right_file"].get<std::string>());
        std::uint64_t ao=item["left_offset"],bo=item["right_offset"];unsigned n=item["vocab"];
        double am=-std::numeric_limits<double>::infinity(),bm=am;
        long double asum=0,bsum=0;
        for(unsigned i=0;i<n;++i) {
            double x=bf16(a.u16(ao+std::uint64_t(i)*2)),y=bf16(b.u16(bo+std::uint64_t(i)*2));
            if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("non-finite observed logits");
            am=std::max(am,x);bm=std::max(bm,y);asum+=x;bsum+=y;
        }
        double ac=double(asum/n),bc=double(bsum/n);long double az=0,bz=0;
        for(unsigned i=0;i<n;++i) {
            az+=std::exp(bf16(a.u16(ao+std::uint64_t(i)*2))-am);
            bz+=std::exp(bf16(b.u16(bo+std::uint64_t(i)*2))-bm);
        }
        double alz=am+std::log(double(az)),blz=bm+std::log(double(bz));
        long double ab=0,ba=0,se=0;double max_error=0;
        for(unsigned i=0;i<n;++i) {
            double x=bf16(a.u16(ao+std::uint64_t(i)*2)),y=bf16(b.u16(bo+std::uint64_t(i)*2));
            double ratio=(x-alz)-(y-blz),error=(x-ac)-(y-bc);
            ab+=std::exp(x-alz)*ratio;ba-=std::exp(y-blz)*ratio;se+=error*error;
            max_error=std::max(max_error,std::abs(error));
        }
        auto [aid,margin_a]=maximum_token(a,ao,n);auto [bid,margin_b]=maximum_token(b,bo,n);
        item["kl_left_right"]=double(ab);item["kl_right_left"]=double(ba);
        item["centered_logit_rmse"]=std::sqrt(double(se/n));item["centered_logit_max_abs"]=max_error;
        item["left_argmax"]=aid;item["right_argmax"]=bid;item["argmax_agreement"]=aid==bid;
        item["left_top2_margin"]=margin_a;item["right_top2_margin"]=margin_b;
        item["vocabulary_rows"]=n;
        if(item.contains("target_token")) {
            unsigned target=item["target_token"];
            if(target>=n)throw std::runtime_error("target outside public vocabulary");
            double x=bf16(a.u16(ao+std::uint64_t(target)*2))-alz;
            double y=bf16(b.u16(bo+std::uint64_t(target)*2))-blz;
            item["left_target_logprob"]=x;item["right_target_logprob"]=y;
            item["target_logprob_difference"]=x-y;
        }
        out<<item.dump()<<'\n'<<std::flush;
    }
}
void acceptance(const json& jobs,const fs::path& output) {
    std::ofstream out(output,std::ios::app);
    for(auto item:jobs) {
        Mapping values(item["file"].get<std::string>());unsigned vocab=item["vocab"];
        std::uint64_t offset=item["offset"];
        auto drafts=item["drafts"].get<std::vector<int>>();
        unsigned extent=item["extent"],budget=item["remaining_budget"],expected=0;
        bool ties=false;
        for(unsigned col=0;col<extent && col+1<budget;++col) {
            auto [id,margin]=maximum_token(values,offset+std::uint64_t(col)*vocab*2,vocab);
            ties|=margin==0;
            if(id!=unsigned(drafts.at(col)))break;
            ++expected;
        }
        auto [bonus,margin]=maximum_token(values,offset+std::uint64_t(expected)*vocab*2,vocab);
        item["expected_accepted"]=expected;item["expected_bonus"]=bonus;item["argmax_tie_seen"]=ties || margin==0;
        item["acceptance_correct"]=expected==item["actual_accepted"].get<unsigned>();
        item["bonus_correct"]=bonus==item["actual_unforced_bonus"].get<unsigned>();
        item["pattern"]=expected==extent?"full":(expected==0?"reject_all":(budget<=extent?"budget_tail":"partial"));
        out<<item.dump()<<'\n'<<std::flush;
    }
}
int main(int argc,char**argv) {
    try {
        if(argc==4) {
            auto jobs=json::parse(std::ifstream(argv[2]));std::string mode=argv[1];
            if(mode=="--distributions")distributions(jobs,argv[3]);
            else if(mode=="--acceptance")acceptance(jobs,argv[3]);
            else throw std::runtime_error("unknown CPU analysis mode");
            return 0;
        }
        if(argc!=3)throw std::runtime_error("usage: precision-oracle manifest.json output.jsonl");
        auto cases=json::parse(std::ifstream(argv[1]));
        for(auto& item:cases) {
            std::cerr<<"ORACLE "<<item["candidate"]<<'\n'<<std::flush;
            evaluate(item["canonical"].get<std::string>(),item["candidate"].get<std::string>(),argv[2]);
        }
    }catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
