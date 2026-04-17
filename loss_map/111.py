import base64
import gzip
from io import BytesIO

def read_varint(data, offset):
    shift = 0
    result = 0
    while True:
        if offset >= len(data):
            raise IndexError("Varint read out of range")
        b = data[offset]
        result |= (b & 0x7F) << shift
        offset += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, offset

def read_length_delimited(data, offset):
    length, offset = read_varint(data, offset)
    value = data[offset:offset+length]
    return value, offset + length

def parse_protobuf(data, depth=0):
    offset = 0
    indent = '  ' * depth
    while offset < len(data):
        try:
            key, offset = read_varint(data, offset)
        except IndexError:
            break
        field_number = key >> 3
        wire_type = key & 0x7

        if wire_type == 0:  # varint
            value, offset = read_varint(data, offset)
            print(f"{indent}[Field {field_number}] varint: {value}")

        elif wire_type == 2:  # length-delimited
            value_bytes, offset = read_length_delimited(data, offset)
            # 先尝试 UTF-8 解码
            try:
                value_str = value_bytes.decode('utf-8')
                print(f"{indent}[Field {field_number}] string: {value_str}")
            except UnicodeDecodeError:
                # 尝试 gzip 解压
                if value_bytes[:2] == b'\x1f\x8b':
                    try:
                        with gzip.GzipFile(fileobj=BytesIO(value_bytes)) as f:
                            decompressed = f.read()
                        print(f"{indent}[Field {field_number}] gzip 解压 {len(decompressed)} bytes")
                        parse_protobuf(decompressed, depth=depth+1)
                    except:
                        print(f"{indent}[Field {field_number}] bytes({len(value_bytes)}): {value_bytes.hex()[:60]}...")
                else:
                    # 当作嵌套 protobuf 递归解析
                    print(f"{indent}[Field {field_number}] length-delimited ({len(value_bytes)} bytes)")
                    parse_protobuf(value_bytes, depth=depth+1)
        else:
            print(f"{indent}[Field {field_number}] Unknown wire type: {wire_type}")
            break

if __name__ == "__main__":
    # 替换为抓到的 Base64 消息
    base64_msg = "CGUQARoOCPiUzuaKMxD4lM7mijMg+JTO5ooz"
    raw_bytes = base64.b64decode(base64_msg)
    print(f"[+] Base64 解码字节长度: {len(raw_bytes)}")
    parse_protobuf(raw_bytes)
