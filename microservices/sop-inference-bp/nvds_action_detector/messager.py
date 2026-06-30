#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import os
import threading
import uuid
from typing import Optional

from confluent_kafka import Consumer, KafkaException, Producer
from confluent_kafka.error import KafkaError
from google.protobuf.json_format import MessageToDict

from . import ds_logger
from .protos import nv_pb2

logger = ds_logger.get_logger(__name__)

DEFAULT_TOPIC = os.getenv("DEFAULT_TOPIC", "mdx-vlm-captions")
SOP_MESSAGING_SCHEMA = os.getenv("SOP_MESSAGING_SCHEMA", "JSON")


def _delivery_report(err, msg):
    if err:
        logger.error(f"Message delivery failed: {err}")


def convert_to_vision_llm(data: dict) -> nv_pb2.VisionLLM:
    msg = nv_pb2.VisionLLM()
    msg.version = "1.0"

    # Timestamps
    start_time = data.get("start_time", 0.0)
    end_time = data.get("end_time", 0.0)

    first_timestamp = data.get("first_timestamp", 0.0)

    start_time = start_time + first_timestamp
    end_time = end_time + first_timestamp

    msg.timestamp.seconds = int(start_time)
    msg.timestamp.nanos = int((start_time - int(start_time)) * 1e9)

    msg.end.seconds = int(end_time)
    msg.end.nanos = int((end_time - int(end_time)) * 1e9)

    msg.startFrameId = "0"
    msg.endFrameId = "0"

    # Sensor
    sensor_id = data.get("sensor_id", "unknown")
    msg.sensor.id = sensor_id
    msg.sensor.type = "Camera"
    msg.sensor.description = data.get("file_path", "")

    # LLM
    query = msg.llm.queries.add()
    query.response = data.get("response", "")
    req_id = data.get("req_id")
    if req_id:
        query.id = str(req_id)

    # Info
    if "chunk_idx" in data:
        msg.info["chunk_idx"] = str(data["chunk_idx"])
    if "cv_execute_time" in data:
        msg.info["cv_execute_time"] = str(data["cv_execute_time"])
    if "vlm_execute_time" in data:
        msg.info["vlm_execute_time"] = str(data["vlm_execute_time"])
    if "frame_number" in data:
        msg.info["frame_number"] = str(data["frame_number"])
    if "checker_execute_time" in data:
        msg.info["checker_execute_time"] = str(data["checker_execute_time"])
    if "first_timestamp" in data:
        msg.info["first_timestamp"] = str(data["first_timestamp"])

    checker_result = data.get("checker_result")
    if checker_result:
        try:
            msg.info["checker_result"] = json.dumps(checker_result)
        except Exception:
            pass

    return msg


class BaseMessageProducer:
    def __init__(self, kafka_broker: str):
        self._kafka_broker = kafka_broker
        self._producer = None

    def _get_client_id(self, pid):
        return f"producer-ds-sop-{pid}"

    def _ensure_producer(self):
        if self._producer is None:
            current_pid = os.getpid()
            conf = {
                "bootstrap.servers": self._kafka_broker,
                "client.id": self._get_client_id(current_pid),
            }
            self._producer = Producer(conf)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_producer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._producer = None

    def produce(self, chunk_info: dict, request_id: Optional[str] = None, topic: str = DEFAULT_TOPIC):
        raise NotImplementedError

    def close(self):
        if self._producer:
            self._producer.flush()


class NvProtoMessageProducer(BaseMessageProducer):
    def _get_client_id(self, pid):
        return f"producer-ds-sop-mp-{pid}"

    def produce(self, chunk_info: dict, request_id: Optional[str] = None, topic: str = DEFAULT_TOPIC):
        self._ensure_producer()
        try:
            vision_llm = convert_to_vision_llm(chunk_info)
            value = vision_llm.SerializeToString()
            sensor_id = chunk_info.get("sensor_id", "unknown_sensor")
            key = str(sensor_id).encode("utf-8")

            self._producer.produce(topic, value=value, key=key, on_delivery=_delivery_report)
            self._producer.poll(0)
        except Exception as e:
            logger.exception(f"Error in produce_chunk: {e}")


class JSONMessageProducer(BaseMessageProducer):
    def __init__(self, kafka_broker: str = "localhost:9092"):
        super().__init__(kafka_broker)

    def produce(self, chunk_info: dict, request_id: Optional[str] = None, topic: str = DEFAULT_TOPIC):
        self._ensure_producer()

        def delivery_report(err, msg):
            if err:
                logger.error(f"chunk messaging delivery failed: {err}")
            else:
                logger.info(f"chunk messaging delivered to {msg.topic()} [{msg.partition()}]")

        if request_id is None:
            request_id = chunk_info.get("request_id", str(uuid.uuid4()))
        try:
            value = json.dumps(chunk_info).encode("utf-8")
            key = str(request_id).encode("utf-8")
            logger.info(f"Producing message: {key} -> {value}")
            self._producer.produce(topic, value=value, key=key, on_delivery=delivery_report)
            self._producer.flush()
        except Exception as e:
            logger.error(f"Error in produce: {e}")
            raise e


def create_producer(kafka_broker: str):
    schema = SOP_MESSAGING_SCHEMA
    if schema == "NvProtoSchema":
        return NvProtoMessageProducer(kafka_broker)
    elif schema == "JSON":
        return JSONMessageProducer(kafka_broker)
    else:
        logger.warning(f"Unknown SOP_MESSAGING_SCHEMA: {schema}, defaulting to NvProtoSchema")
        return NvProtoMessageProducer(kafka_broker)


class BaseMessageConsumer:
    def __init__(self, kafka_broker: str = "localhost:9092", group_id: str = "my-group"):
        self._kafka_broker = kafka_broker
        conf = {
            "bootstrap.servers": self._kafka_broker,  # Use the parameter, not hardcoded localhost
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        }
        self._consumer = Consumer(conf)
        self._stop_event = threading.Event()
        self._thread = None

    def _decode_value(self, msg_value):
        raise NotImplementedError

    def consume(self, topic: str = DEFAULT_TOPIC):
        self._consumer.subscribe([topic])
        while not self._stop_event.is_set():
            msg = self._consumer.poll(1.0)
            if msg is None:
                continue
            elif msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"Reached end of partition at offset {msg.offset()}")
                    continue
                else:
                    raise KafkaException(msg.error())
            else:
                key = msg.key().decode("utf-8") if msg.key() else None
                try:
                    value = self._decode_value(msg.value())
                except Exception as e:
                    logger.error(f"Failed to decode message: {e}")
                    continue
                # logger.info(f"Received message: {key} -> {value}")
                yield key, value
        return None

    def start_thread(self):
        logger.info(f"Consumer starting thread")

        def consume_thread():
            for msg in self.consume():
                if msg is None:
                    break
                try:
                    key, value = msg
                    if key is not None and value is not None:
                        dump_json = json.dumps(value, indent=2)
                        logger.info(f"Received message: {key} ->\n{dump_json}")
                except Exception as e:
                    logger.exception(f"Error in consume_thread: {e}")
                    continue

        self._thread = threading.Thread(target=consume_thread, daemon=True)
        self._thread.start()

    def stop_thread(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)  # Wait max 5 seconds for graceful shutdown

    def wait_for_thread(self):
        if self._thread is not None:
            self._thread.join()

    def close(self):
        self._consumer.close()


class NvProtoMessageConsumer(BaseMessageConsumer):
    def _decode_value(self, msg_value):
        vision_llm = nv_pb2.VisionLLM()
        vision_llm.ParseFromString(msg_value)
        return MessageToDict(vision_llm)


class JSONMessageConsumer(BaseMessageConsumer):
    def _decode_value(self, msg_value):
        return json.loads(msg_value)


def create_consumer(kafka_broker: str, group_id: str):
    schema = SOP_MESSAGING_SCHEMA
    if schema == "NvProtoSchema":
        return NvProtoMessageConsumer(kafka_broker, group_id)
    elif schema == "JSON":
        return JSONMessageConsumer(kafka_broker, group_id)
    else:
        logger.warning(f"Unknown SOP_MESSAGING_SCHEMA: {schema}, defaulting to NvProtoSchema")
        return NvProtoMessageConsumer(kafka_broker, group_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Message Producer and Consumer")
    parser.add_argument("--producer", action="store_true", help="Run as producer")
    parser.add_argument("--consumer", action="store_true", help="Run as consumer")
    parser.add_argument("--kafka-broker", type=str, default="localhost:9092", help="Kafka broker")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Topic to produce and consume")
    parser.add_argument(
        "--group-id", type=str, default=None, help="Consumer group ID (defaults to unique ID for testing)"
    )
    args = parser.parse_args()

    if args.producer:
        producer = create_producer(args.kafka_broker)
        from .sop_step_checker import SopCheckerResponse

        response = SopCheckerResponse(
            request_id="123",
            error_message="",
            checker_id="",
            cycle=0,
            missing_detected=[],
            misordered_detected=[],
            final_missing_detected=[],
            final_misordered_detected=[],
            cycle_completed=False,
            summary_cycles_detected=[],
            summary_cycle_analysis=[],
        )
        chunk_data = {
            "req_id": "123",
            "checker_result": response.asdict(),
            "start_time": 0.0,
            "end_time": 1.0,
        }
        producer.produce(chunk_data)
        producer.close()
    if args.consumer:
        # Use unique group ID for testing to avoid partition assignment conflicts
        group_id = args.group_id if args.group_id else f"test-consumer-{uuid.uuid4()}"
        consumer = create_consumer(args.kafka_broker, group_id=group_id)
        logger.info(f"Consumer started with group_id: {group_id}")
        consumer.start_thread()
        import signal
        import sys

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            consumer.stop_thread()
            consumer.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            consumer.wait_for_thread()
        finally:
            consumer.close()
