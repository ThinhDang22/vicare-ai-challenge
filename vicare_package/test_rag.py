"""Kiểm thử nhanh phần logic thuần (không cần GPU/mạng):  python test_rag.py"""
from rag import EMERGENCY, REFUSAL, chunk_text, format_answer, rrf_fuse, tokenize

assert chunk_text("a b c d e f g", 4, 1) == ["a b c d", "d e f g"]
assert tokenize("Viêm gan B") == ["viêm", "gan", "b", "viêm_gan", "gan_b"]
s = rrf_fuse([["x", "y"], ["y", "z"]])
assert max(s, key=s.get) == "y"
assert format_answer(REFUSAL, "Người nhà bất tỉnh").startswith(EMERGENCY)
assert not format_answer(REFUSAL, "Cách nấu phở bò").startswith(EMERGENCY)
print("OK")
