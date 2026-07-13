# Data migration manifest
Generated: 2026-07-13T04:06:29Z (from DA_GestureRecognition, pre-migration)
Host: Alien-2051

## Data_Processed/ntu_quats (IRREPLACEABLE -- raw NTU-SkeletalData no longer exists on this machine)
Files: 43490
Size: 976M
SHA256 count: 43490

## DataCollection/ (IRREPLACEABLE -- raw Xsens mvnx recordings, PII in raw metadata, see dataset_pii_hazard)
Files: 64
Size: 6.8G
SHA256 count: 64

## external_data/ (replaceable in principle -- public CZU-MHAD + UTD-MHAD downloads; size/count only, no checksums)
    8.2G	external_data
    4.1G	external_data/czu_mhad_data
    76M	external_data/czu_mhad
    31M	external_data/utd_mhad
Total files: 8022

## Data_Processed/ (regeneratable from the above via existing parsers, listed for reference only)
    1.6G	Data_Processed
    976M	Data_Processed/ntu_quats
    68M	Data_Processed/imu_quats_v2
    68M	Data_Processed/imu_quats_alpha2.00
    68M	Data_Processed/imu_quats_alpha1.50
    68M	Data_Processed/imu_quats_alpha0.75
    68M	Data_Processed/imu_quats_alpha0.50
    68M	Data_Processed/imu_quats_alpha0.25
    68M	Data_Processed/imu_quats
    41M	Data_Processed/czu_skeleton_lrq
    33M	Data_Processed/czu_imu_quats
    32M	Data_Processed/czu_imu_raw
    18M	Data_Processed/utd_skeleton_lrq
    13M	Data_Processed/czu_imu_mag20

## Explicitly NOT migrated (left behind in the old repo)
- trained_models/ (234GB) -- entire clean rerun starts empty, nothing carried over, per plan
- .venv/ (9.6GB) -- rebuilt fresh from requirements.txt in the new repo
- temp_outputs/ (3.1GB) -- confirmed dead end (abandoned slerp-cleaning experiment, see [[cleaned-source-pretraining]])
- .git/ (1.8GB history) -- fresh git init in the new repo, old history stays here for reference
- .obsidian/ -- personal vault config, not project data

## Migration status: DONE (2026-07-13)
- ntu_quats: copied via rsync, verified byte-identical against ntu_quats.sha256 (43490/43490 files, exit 0)
- DataCollection: copied via rsync, verified byte-identical against DataCollection.sha256 (64/64 files, exit 0)
- external_data: copied via rsync, size-verified only (8.3G, no checksums per plan)
- Originals in the old repo (DA_GestureRecognition) were NOT deleted -- both copies exist until archived/confirmed by the human.
