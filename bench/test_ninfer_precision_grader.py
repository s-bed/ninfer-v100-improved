"""Execution-environment checks against valid pure-Python task solutions."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import ninfer_precision as campaign
from ninfer_precision_grade import grade


class GraderEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={case['id']:case for case in campaign.quality_cases()}

    def test_order_preserving_dict_fromkeys_is_available(self):
        result=grade(self.cases['code-002'],'def solve_2(xs):\n    return list(dict.fromkeys(xs))')
        self.assertTrue(result['correct'],result)

    def test_ordered_dict_fromkeys_is_available(self):
        result=grade(self.cases['code-002'],
            'from collections import OrderedDict\ndef solve_2(xs):\n    return list(OrderedDict.fromkeys(xs))')
        self.assertTrue(result['correct'],result)

    def test_regex_module_can_normalize_palindrome(self):
        result=grade(self.cases['code-001'],
            'import re\ndef solve_1(text):\n    text = re.sub(r"[\\W_]", "", text).casefold()\n    return text == text[::-1]')
        self.assertTrue(result['correct'],result)

    def test_compiled_regex_is_not_python_dynamic_code_compilation(self):
        result=grade(self.cases['code-001'],
            'import re\ndef solve_1(text):\n    pattern = re.compile(r"[\\W_]", re.IGNORECASE)\n    text = pattern.sub("", text).casefold()\n    return text == text[::-1]')
        self.assertTrue(result['correct'],result)

    def test_pure_regex_import_member_alias_works(self):
        result=grade(self.cases['code-001'],
            'from re import sub as regex_sub\ndef solve_1(text):\n    text = regex_sub(r"[\\W_]", "", text).casefold()\n    return text == text[::-1]')
        self.assertTrue(result['correct'],result)

    def test_direct_regex_compile_import_is_available(self):
        result=grade(self.cases['code-001'],
            'from re import compile\ndef solve_1(text):\n    text = compile(r"[\\W_]").sub("", text).casefold()\n    return text == text[::-1]')
        self.assertTrue(result['correct'],result)

    def test_math_inf_is_available_for_nonempty_maximum_sum(self):
        result=grade(self.cases['code-007'],
            'import math\ndef solve_7(xs):\n    if not xs: return 0\n    best = -math.inf\n    total = 0\n    for value in xs:\n        total = max(value, total + value)\n        best = max(best, total)\n    return best')
        self.assertTrue(result['correct'],result)

    def test_forbidden_capabilities_remain_unavailable(self):
        snippets=[
            'import os\ndef solve_2(xs): return os.listdir("/")',
            'import sys\ndef solve_2(xs): return sys.modules',
            'from re import __dict__\ndef solve_2(xs): return []',
            'import re\ndef solve_2(xs): return re.__dict__',
            'def solve_2(xs): return eval("xs")',
            'def solve_2(xs): return exec("pass")',
            'def solve_2(xs): return compile("pass", "x", "exec")',
            'def solve_2(xs): return open("/etc/passwd").read()',
            'def solve_2(xs): return getattr(xs, "__class__")',
            'def solve_2(xs): return globals()',
            'def solve_2(xs): return xs.__class__',
            'import re\ndef solve_2(xs): return re.compile.__globals__',
        ]
        for source in snippets:
            with self.subTest(source=source):self.assertFalse(grade(self.cases['code-002'],source)['correct'])

    def test_environment_fix_does_not_accept_incorrect_results(self):
        for cid,source in [
            ('code-002','def solve_2(xs): return []'),
            ('code-001','import re\ndef solve_1(text): return re.sub(r"[\\W_]", "", text)'),
            ('code-007','import math\ndef solve_7(xs): return 0'),
        ]:
            with self.subTest(case=cid):self.assertFalse(grade(self.cases[cid],source)['correct'])


if __name__=='__main__':unittest.main()
