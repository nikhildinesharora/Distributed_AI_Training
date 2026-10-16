import asyncio
import unittest

from dml_cluster.protocol import (
    ProtocolError,
    read_binary,
    read_message,
    send_binary,
    send_message,
)


class MemoryWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def write(self, payload: bytes) -> None:
        self.buffer.extend(payload)

    async def drain(self) -> None:
        return None


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_message_round_trip(self) -> None:
        writer = MemoryWriter()
        await send_message(writer, {"type": "heartbeat", "rank": 2})

        reader = asyncio.StreamReader()
        reader.feed_data(bytes(writer.buffer))
        reader.feed_eof()

        self.assertEqual(
            await read_message(reader),
            {"rank": 2, "type": "heartbeat"},
        )

    async def test_non_object_message_is_rejected(self) -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b"[1,2,3]\n")
        reader.feed_eof()

        with self.assertRaises(ProtocolError):
            await read_message(reader)

    async def test_binary_payload_round_trip(self) -> None:
        writer = MemoryWriter()
        payload = b"model-shard"
        await send_binary(writer, payload)

        reader = asyncio.StreamReader()
        reader.feed_data(bytes(writer.buffer))
        reader.feed_eof()

        self.assertEqual(await read_binary(reader, expected_size=len(payload)), payload)

    async def test_binary_size_mismatch_is_rejected(self) -> None:
        writer = MemoryWriter()
        await send_binary(writer, b"abc")

        reader = asyncio.StreamReader()
        reader.feed_data(bytes(writer.buffer))
        reader.feed_eof()

        with self.assertRaises(ProtocolError):
            await read_binary(reader, expected_size=4)


if __name__ == "__main__":
    unittest.main()
