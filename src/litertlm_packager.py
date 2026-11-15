import struct
from pathlib import Path
from typing import Any, Dict, List

LITERTLM_HEADER_368_HEX = (
    "4c49544552544c4d010000000500000000000000000000007001000000000000"
    "0c00000008000c00080004000800000008000000ec000000eafeffff04000000"
    "03000000bc00000088000000040000005cffffff00000003f0cef31000000000"
    "004048000000000004000000010000001000000000000a0010000c000b000400"
    "0a0000000c000000000000092800000042ffffff040000001600000074665f6c"
    "6974655f70726566696c6c5f6465636f646500000a0000006d6f64656c5f7479"
    "706500000c0020001c001000080007000c00000000000004f80c480000000000"
    "00800000000000000000000004000000000000000c001c001800100008000700"
    "0c000000000000053c4000000000000000400000000000000400000000000000"
    "d2ffffff04000000010000001000000000000a0012000c000b0004000a000000"
    "14000000000000091c0000000000060008000400060000000400000000000000"
    "0000000007000000417574686f727300"
)

DEFAULT_LLM_METADATA_PROTO_HEX = (
    "0a050a030a0102120f120d3c656e645f6f665f7475726e3e"
    "121b12193c73746172745f66756e6374696f6e5f726573706f6e73653e28800832021a00"
)

SECTION_DATA_TYPES = [
    "NONE",
    "GenericBinaryData",
    "Deprecated",
    "TFLiteModel",
    "SP_Tokenizer",
    "LlmMetadataProto",
    "HF_Tokenizer_Zlib",
    "TFLiteWeights",
    "EmbeddingMetadataProto",
    "ExecutorMetadataProto",
    "TtsMetadataProto",
    "AsrMetadataProto",
]


def _align_16k(offset: int) -> int:
    return (offset + 16383) & ~16383


def build_litertlm_container(
    output_path: Path,
    sp_tokenizer_bytes: bytes,
    tflite_model_bytes: bytes,
    llm_metadata_proto_bytes: bytes | None = None,
) -> Dict[str, Any]:
    """Builds a valid LiteRT-LM (.litertlm v1.5.0) container file matching flutter_edge_ai_litertlm."""
    if llm_metadata_proto_bytes is None:
        llm_metadata_proto_bytes = bytes.fromhex(DEFAULT_LLM_METADATA_PROTO_HEX)

    header_buf = bytearray(bytes.fromhex(LITERTLM_HEADER_368_HEX))

    sec0_begin = 16384
    sec0_end = sec0_begin + len(llm_metadata_proto_bytes)

    sec1_begin = 32768
    sec1_end = sec1_begin + len(sp_tokenizer_bytes)

    sec2_begin = _align_16k(sec1_end)
    sec2_end = sec2_begin + len(tflite_model_bytes)

    # Patch exact uint64 offsets in the FlatBuffer header
    # Section 0 (LlmMetadataProto): end at 264, begin at 272
    struct.pack_into("<Q", header_buf, 264, sec0_end)
    struct.pack_into("<Q", header_buf, 272, sec0_begin)

    # Section 1 (SP_Tokenizer): end at 216, begin at 224
    struct.pack_into("<Q", header_buf, 216, sec1_end)
    struct.pack_into("<Q", header_buf, 224, sec1_begin)

    # Section 2 (TFLiteModel tf_lite_prefill_decode): end at 88, begin at 96
    struct.pack_into("<Q", header_buf, 88, sec2_end)
    struct.pack_into("<Q", header_buf, 96, sec2_begin)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header_buf)
        f.write(b"\x00" * (sec0_begin - len(header_buf)))
        f.write(llm_metadata_proto_bytes)
        f.write(b"\x00" * (sec1_begin - sec0_end))
        f.write(sp_tokenizer_bytes)
        f.write(b"\x00" * (sec2_begin - sec1_end))
        f.write(tflite_model_bytes)

    return verify_litertlm_container(output_path)


def verify_litertlm_container(file_path: Path) -> Dict[str, Any]:
    """Parses and validates a .litertlm container using the exact flutter_edge_ai_litertlm rules."""
    file_size = file_path.stat().st_size
    with open(file_path, "rb") as f:
        preamble = f.read(32)
        if len(preamble) < 32 or preamble[:8] != b"LITERTLM":
            raise ValueError("Not a .litertlm file: missing LITERTLM magic")
        major, minor, patch, _pad, header_end = struct.unpack("<IIIIQ", preamble[8:32])
        if major != 1:
            raise ValueError(f"Unsupported .litertlm major version: {major}")
        header = f.read(header_end - 32)

    def u32(p: int) -> int:
        return struct.unpack_from("<I", header, p)[0]

    def i32(p: int) -> int:
        return struct.unpack_from("<i", header, p)[0]

    def u16(p: int) -> int:
        return struct.unpack_from("<H", header, p)[0]

    def u8(p: int) -> int:
        return struct.unpack_from("<B", header, p)[0]

    def u64(p: int) -> int:
        return struct.unpack_from("<Q", header, p)[0]

    def ref(p: int) -> int:
        return p + u32(p)

    def field(t: int, idx: int) -> int | None:
        vt = t - i32(t)
        vts = u16(vt)
        slot = 4 + 2 * idx
        if slot >= vts:
            return None
        off = u16(vt + slot)
        return None if off == 0 else t + off

    def table(t: int, idx: int) -> int | None:
        at = field(t, idx)
        return None if at is None else ref(at)

    root = ref(0)
    sm = table(root, 1)
    objs = table(sm, 0) if sm is not None else None
    sections: List[Dict[str, Any]] = []
    if objs is not None:
        n = u32(objs)
        for i in range(n):
            sec = ref(objs + 4 + 4 * i)
            f_dt = field(sec, 3)
            f_b = field(sec, 1)
            f_e = field(sec, 2)
            dt = u8(f_dt) if f_dt is not None else 0
            b = u64(f_b) if f_b is not None else 0
            e = u64(f_e) if f_e is not None else 0
            sections.append(
                {
                    "index": i,
                    "data_type": SECTION_DATA_TYPES[dt] if dt < len(SECTION_DATA_TYPES) else f"unknown({dt})",
                    "begin": b,
                    "end": e,
                    "length": e - b,
                }
            )

    expected_end = max((s["end"] for s in sections), default=0)
    return {
        "valid": True,
        "version": f"{major}.{minor}.{patch}",
        "header_end": header_end,
        "file_length": file_size,
        "expected_length": expected_end,
        "is_complete": file_size >= expected_end,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "sections": sections,
    }
