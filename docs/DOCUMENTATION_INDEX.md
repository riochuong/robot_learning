# Documentation Index

Quick reference to all documentation files in this project.

---

## 📚 Core Knowledge Base

### **[LEROBOT_DATASET_KNOWLEDGE_BASE.md](LEROBOT_DATASET_KNOWLEDGE_BASE.md)** ⭐ START HERE
> **The master reference for everything LeRobot datasets**
> 
> Comprehensive guide covering:
> - Dataset structure and video storage architecture
> - How LeRobot links videos to episodes (episode metadata)
> - Recording, verification, and debugging
> - All common issues with solutions
> - Tools reference and best practices
> - Troubleshooting checklists
> - Quick reference commands
>
> **Use this as context in future AI sessions!**

---

## 🔧 Tools and Usage

### **[README.md](README.md)**
> Quick command reference for common tasks:
> - Recording with cameras
> - Calibration testing
> - Dataset viewing
> - Teleoperation

### **[VIEWER_USAGE.md](VIEWER_USAGE.md)**
> Detailed guide for `view_dataset_local.py`:
> - How to browse episodes
> - Keyboard controls
> - Quality checking workflow
> - Tips and best practices

### **[DATASET_SAFETY.md](DATASET_SAFETY.md)**
> Assurance that `view_dataset_local.py` is 100% read-only
> - What the script does
> - What it doesn't do
> - Technical proof

---

## 🎯 Specific Topics

### **[CALIBRATION_TEST_LOGIC.md](CALIBRATION_TEST_LOGIC.md)**
> Deep dive into `test_calibration.py`:
> - Purpose and workflow
> - Function-by-function explanation
> - Key concepts (observation processing, timing)
> - Data flow diagrams

### **[DATASET_VERIFICATION_SUMMARY.md](DATASET_VERIFICATION_SUMMARY.md)**
> Example verification report for `dataset/pick_and_place_small_cube`:
> - What a valid dataset looks like
> - Metadata structure verification
> - Episode and video mappings
> - Reference for comparison

### **[CORRECTION_DATA_IS_VALID.md](CORRECTION_DATA_IS_VALID.md)**
> Important correction about video storage:
> - Why "missing video columns" is NOT a bug
> - How LeRobot actually stores videos
> - Episode metadata explanation
> - Lessons learned from misdiagnosis

### **[DATASET_EDITING_GUIDE.md](DATASET_EDITING_GUIDE.md)** ⭐ NEW
> Complete guide to editing datasets with LeRobot's built-in tools:
> - Delete bad episodes from recordings
> - Split datasets into train/val/test sets
> - Merge multiple recording sessions
> - Remove cameras/features to save space
> - Convert old image datasets to video format
> - Python API and CLI examples
> - Common use cases and troubleshooting

---

## 🛠️ Tools (Scripts)

### **[verify_dataset.py](verify_dataset.py)** ⭐ ESSENTIAL
> Comprehensive dataset verification tool
> 
> ```bash
> uv run python verify_dataset.py dataset/my_dataset
> ```
> 
> Checks:
> - Directory structure
> - Metadata files
> - Episode counts
> - Video mappings
> - Data validity
> 
> Output: Pass/fail verdict with detailed report

### **[view_dataset_local.py](view_dataset_local.py)** ⭐ ESSENTIAL
> Custom dataset viewer for local-only datasets
> 
> ```bash
> uv run python view_dataset_local.py data/my_dataset --all
> ```
> 
> Features:
> - Works without HuggingFace Hub
> - Accurate video loading via episode metadata
> - Fresh Rerun window per episode
> - Browse all episodes sequentially

### **[test_calibration.py](test_calibration.py)**
> Robot calibration testing script
> 
> ```bash
> uv run python test_calibration.py \
>     --robot.type=so101_follower \
>     --dataset.test_episodes=0,1,2
> ```
> 
> Records and replays trajectories to verify calibration accuracy

### **[inspect_parquet.py](inspect_parquet.py)**
> Parquet file inspector
> 
> ```bash
> uv run python inspect_parquet.py path/to/file.parquet
> ```
> 
> Shows schema, statistics, and sample rows

### **[camera_test.py](camera_test.py)**
> Camera functionality tester
> 
> ```bash
> uv run python camera_test.py 2 4
> ```
> 
> Tests multiple cameras and saves sample images

### **[view_episodes.sh](view_episodes.sh)**
> Simple wrapper for `view_dataset_local.py`
> 
> ```bash
> ./view_episodes.sh data/my_dataset 0
> ```

---

## 📖 How to Use This Documentation

### For New Users:
1. Read **LEROBOT_DATASET_KNOWLEDGE_BASE.md** (sections 1-4)
2. Try **verify_dataset.py** on example datasets
3. Use **README.md** for quick commands
4. Refer back to knowledge base when issues arise

### For Debugging:
1. Check **LEROBOT_DATASET_KNOWLEDGE_BASE.md** section 5 (Common Issues)
2. Use troubleshooting checklist in section 10
3. Run **verify_dataset.py** for automated checks
4. Inspect files with **inspect_parquet.py**
5. View episodes with **view_dataset_local.py**

### For Recording:
1. Review **LEROBOT_DATASET_KNOWLEDGE_BASE.md** section 3 (Recording)
2. Copy command from **README.md**
3. Verify after recording with **verify_dataset.py**
4. Check quality with **view_dataset_local.py --all**

### For Training:
1. Verify dataset with **verify_dataset.py**
2. Check first and last episodes with **view_dataset_local.py**
3. Review best practices in **LEROBOT_DATASET_KNOWLEDGE_BASE.md** section 8
4. Filter bad episodes if needed

### For AI Context:
Include **LEROBOT_DATASET_KNOWLEDGE_BASE.md** in context for:
- Future debugging sessions
- Dataset-related questions
- Tool development
- Team onboarding

---

## 📊 Quick Stats

- **Documentation files:** 7
- **Tool scripts:** 6
- **Total issues documented:** 6 with full solutions
- **Code examples:** 15+
- **Commands ready to use:** 20+

---

## 🔄 Update History

- **Dec 14, 2025:** Initial documentation index created
  - Comprehensive knowledge base compiled
  - All learnings from dataset debugging documented
  - Tools created and documented

---

## 💡 Pro Tips

1. **Always start with LEROBOT_DATASET_KNOWLEDGE_BASE.md** - it has everything
2. **Keep README.md updated** with working commands for your setup
3. **Run verify_dataset.py after every recording** - catch issues early
4. **Use --json flag** with verify_dataset.py for scripting
5. **Add this index to git** - helps future you and collaborators

---

**Need help?** Start with the knowledge base, check the troubleshooting checklist, then use the tools!

