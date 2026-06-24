import time
import threading
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from scapy.all import sniff, IP, TCP, UDP, ICMP
from sklearn.ensemble import IsolationForest

from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

packet_count = 0
byte_count = 0

protocol_counter = defaultdict(int)
src_counter = defaultdict(int)
dst_counter = defaultdict(int)

feature_window = deque(maxlen=200)

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

trained = False

lock = threading.Lock()


def packet_handler(packet):
    global packet_count
    global byte_count

    with lock:

        packet_count += 1
        byte_count += len(packet)

        if IP in packet:

            src_counter[packet[IP].src] += 1
            dst_counter[packet[IP].dst] += 1

            if TCP in packet:
                protocol_counter["TCP"] += 1

            elif UDP in packet:
                protocol_counter["UDP"] += 1

            elif ICMP in packet:
                protocol_counter["ICMP"] += 1

            else:
                protocol_counter["OTHER"] += 1


def collect_features():

    global packet_count
    global byte_count
    global trained

    prev_packets = 0
    prev_bytes = 0

    while True:

        time.sleep(5)

        with lock:

            pps = packet_count - prev_packets
            bps = byte_count - prev_bytes

            prev_packets = packet_count
            prev_bytes = byte_count

            tcp = protocol_counter["TCP"]
            udp = protocol_counter["UDP"]
            icmp = protocol_counter["ICMP"]

            unique_src = len(src_counter)
            unique_dst = len(dst_counter)

            feature = [
                pps,
                bps,
                tcp,
                udp,
                icmp,
                unique_src,
                unique_dst
            ]

            feature_window.append(feature)

            if len(feature_window) > 30:

                X = np.array(feature_window)

                if not trained:
                    model.fit(X)
                    trained = True

                prediction = model.predict([feature])[0]

                score = model.decision_function([feature])[0]

                if prediction == -1:

                    console.print(
                        f"[red]ALERT[/red] "
                        f"Possible anomaly detected "
                        f"(score={score:.4f})"
                    )

                save_record(feature, prediction, score)


def save_record(feature, prediction, score):

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pps": feature[0],
        "bytes": feature[1],
        "tcp": feature[2],
        "udp": feature[3],
        "icmp": feature[4],
        "unique_src": feature[5],
        "unique_dst": feature[6],
        "anomaly": prediction,
        "score": score
    }

    pd.DataFrame([row]).to_csv(
        "network_log.csv",
        mode="a",
        index=False,
        header=not pd.io.common.file_exists(
            "network_log.csv"
        )
    )


def build_dashboard():

    table = Table(title="Network Anomaly Detector")

    table.add_column("Metric")
    table.add_column("Value")

    with lock:

        table.add_row(
            "Packets",
            str(packet_count)
        )

        table.add_row(
            "Bytes",
            str(byte_count)
        )

        table.add_row(
            "TCP",
            str(protocol_counter["TCP"])
        )

        table.add_row(
            "UDP",
            str(protocol_counter["UDP"])
        )

        table.add_row(
            "ICMP",
            str(protocol_counter["ICMP"])
        )

        table.add_row(
            "Unique Sources",
            str(len(src_counter))
        )

        table.add_row(
            "Unique Destinations",
            str(len(dst_counter))
        )

    return table


def dashboard_loop():

    with Live(
        build_dashboard(),
        refresh_per_second=1
    ) as live:

        while True:

            live.update(
                build_dashboard()
            )

            time.sleep(1)


def main():

    console.print(
        "[green]Starting Network Anomaly Detector[/green]"
    )

    threading.Thread(
        target=collect_features,
        daemon=True
    ).start()

    threading.Thread(
        target=dashboard_loop,
        daemon=True
    ).start()

    sniff(
        prn=packet_handler,
        store=False
    )


if __name__ == "__main__":
    main()
