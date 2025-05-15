from kafka import KafkaConsumer
import json
import base64
import numpy as np
import torch


def deserialize_tensor(message):
    """
    Deserializes a JSON message back into a PyTorch tensor.
    """
    data_b64 = message["data"]
    shape = tuple(message["shape"])
    dtype = message["dtype"]
    data_bytes = base64.b64decode(data_b64)
    arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return torch.tensor(arr)


def forgiving_deserializer(m):
    if not m:
        return None
    try:
        return json.loads(m.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"Bad JSON at offset: {e}")
        return None


consumer = KafkaConsumer(
    "spectrogram",
    bootstrap_servers="localhost:9092",
    group_id="spectrogram_consumers",
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    value_deserializer=forgiving_deserializer,
)


def main():
    print("Starting Kafka consumer...")
    try:
        for msg in consumer:
            message = msg.value
            if message is None:
                continue
            try:
                fbank_tensor = deserialize_tensor(msg.value)
                print("Received tensor with shape:", fbank_tensor.shape)
                consumer.commit()
            except Exception as e:
                print(f"Failed to deserialize message at offset {msg.offset}: {e}")
                continue
    except KeyboardInterrupt:
        print("Consumer interrupted. Exiting...")
    finally:
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
