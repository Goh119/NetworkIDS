from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


EXPECTED_ROOTS = (Path('data/processed'), Path('data/test'))
OUTPUT_PATH = Path('data/dataset_manifest.csv')


def classify_flow(path: Path, split: str) -> dict[str, str]:
    name = path.name
    lower = name.lower()

    row = {
        'flow_csv': str(path).replace('\\', '/'),
        'matching_packet_csv': '',
        'split_folder': split,
        'intended_use': 'REVIEW_REQUIRED',
        'attack_family': 'Unknown',
        'attack_variant': 'Unknown',
        'row_labels_status': 'NOT_ASSIGNED',
        'notes': 'File-level manifest only; this script does not assign labels to individual flows.',
    }

    stem = name[:-len('_flow.csv')] if name.endswith('_flow.csv') else path.stem
    packet_path = path.with_name(f'{stem}_packets.csv')
    row['matching_packet_csv'] = str(packet_path).replace('\\', '/')

    if split == 'processed':
        if lower == 'benign_whole-network3_flow.csv':
            row.update(
                intended_use='TRAIN_CANDIDATE',
                attack_family='Benign',
                attack_variant='Whole-network benign capture candidate',
                notes='Intended normal-data candidate. Verify the source capture/ground truth before assigning row labels.',
            )
        elif lower.startswith('attack_bruteforce_dictionary-'):
            variant = 'SSH dictionary brute force' if '-ssh_' in lower else (
                'Telnet dictionary brute force' if '-telnet_' in lower else 'Dictionary brute force'
            )
            row.update(
                intended_use='TRAIN_CANDIDATE',
                attack_family='Bruteforce',
                attack_variant=variant,
                notes='Training candidate per project plan; do not automatically label every flow as attack without verifying capture ground truth.',
            )
        elif lower.startswith('attack_recon_'):
            variant = name.removeprefix('attack_recon_').removesuffix('_flow.csv')
            row.update(
                intended_use='TRAIN_CANDIDATE',
                attack_family='Recon',
                attack_variant=variant,
                notes='Training candidate per project plan; whole-network captures may contain background flows, so verify row-level ground truth.',
            )
        elif lower.startswith('mqtt-') and lower.endswith('_train_flow.csv'):
            variant = name.removeprefix('MQTT-').removesuffix('_train_flow.csv')
            row.update(
                intended_use='TRAIN_CANDIDATE',
                attack_family='MQTT DoS/DDoS',
                attack_variant=variant,
                notes='Training candidate per project plan; verify source dataset labels and flow mapping before assigning row labels.',
            )
        else:
            row['notes'] = 'Top-level processed flow CSV not recognized by the current project manifest rules; review manually.'

    elif split == 'test':
        if lower.startswith('attack_malware_mirai-'):
            variant = 'SYN flood' if '-syn-flood_' in lower else ('UDP flood' if '-udp-flood_' in lower else 'Mirai variant to review')
            row.update(
                intended_use='PRIMARY_UNSEEN_EVALUATION',
                attack_family='Mirai',
                attack_variant=variant,
                notes='Held out from RF training by project design. Row-level labels still require verified ground truth.',
            )
        elif lower.startswith('mqtt-malformed_data_'):
            variant = 'Malformed data (' + ('source filename says train' if '_train_flow.csv' in lower else 'source filename says test') + ')'
            row.update(
                intended_use='PRIMARY_UNSEEN_EVALUATION',
                attack_family='MQTT Malformed',
                attack_variant=variant,
                notes='Held out from RF training by project design, including the file whose source name contains _train.',
            )
        elif lower.startswith('attack_mitm_impersonation_'):
            variant = name.removeprefix('attack_mitm_impersonation_').removesuffix('_flow.csv')
            row.update(
                intended_use='PRIMARY_UNSEEN_EVALUATION',
                attack_family='MITM Impersonation',
                attack_variant=variant,
                notes='Held out from RF training by project design. Row-level labels still require verified ground truth.',
            )
        elif lower.startswith('attack_mitm_ip-spoofing_'):
            variant = name.removeprefix('attack_mitm_ip-spoofing_').removesuffix('_flow.csv')
            row.update(
                intended_use='SUPPLEMENTARY_UNSEEN_EVALUATION',
                attack_family='IP Spoofing',
                attack_variant=variant,
                notes='Supplementary held-out evaluation; do not use for RF training.',
            )
        elif lower.startswith('mqtt-') and lower.endswith('_test_flow.csv'):
            variant = name.removeprefix('MQTT-').removesuffix('_test_flow.csv')
            row.update(
                intended_use='KNOWN_ATTACK_GENERALIZATION_TEST',
                attack_family='MQTT DoS/DDoS',
                attack_variant=variant,
                notes='Held-out samples from attack variants also represented in training; do not train on these test files.',
            )
        else:
            row['notes'] = 'Top-level test flow CSV not recognized by the current project manifest rules; review manually.'

    return row


def main() -> int:
    cwd = Path.cwd()
    if not (cwd / 'src' / 'packet_parser.py').exists() or not (cwd / 'src' / 'flow_builder.py').exists():
        print('ERROR: Run this script from C:\\FYP\\NetworkIDS (the project root).', file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    missing_dirs = []
    for directory in EXPECTED_ROOTS:
        if not directory.is_dir():
            missing_dirs.append(str(directory))
            continue
        split = directory.name
        for flow_path in sorted(directory.glob('*_flow.csv')):
            rows.append(classify_flow(flow_path, split))

    if missing_dirs:
        print('ERROR: Required folder(s) missing: ' + ', '.join(missing_dirs), file=sys.stderr)
        return 2

    if not rows:
        print('ERROR: No top-level *_flow.csv files found in data\\processed or data\\test.', file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'flow_csv',
        'matching_packet_csv',
        'split_folder',
        'intended_use',
        'attack_family',
        'attack_variant',
        'row_labels_status',
        'notes',
    ]

    with OUTPUT_PATH.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row['split_folder'], row['intended_use']) for row in rows)
    print('=' * 72)
    print('NetworkIDS - Dataset Manifest Created')
    print(f'Project root : {cwd}')
    print(f'Output       : {OUTPUT_PATH}')
    print(f'Flow files   : {len(rows)}')
    print('')
    for (split, usage), count in sorted(counts.items()):
        print(f'{split:10} | {usage:38} | {count:>3}')
    print('')
    print('IMPORTANT: This is a file-level manifest only.')
    print('No flow-row labels were assigned, and no feature CSVs were modified.')
    print('Confirm row-level ground truth before training or reporting metrics.')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
