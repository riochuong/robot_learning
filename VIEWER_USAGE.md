# Dataset Viewer Usage Guide

## How the Viewer Works

### Rerun Viewer Process
When you run `view_dataset_local.py`, it:

1. **Spawns fresh Rerun viewer** - Opens new window for each episode
2. **Loads episode data** - Streams camera feeds and robot data to viewer
3. **Viewer displays data** - You can review at your own pace
4. **You close when done** - Close the Rerun window when finished reviewing
5. **Script waits** - Returns to terminal prompt
6. **Next episode** - Press Enter to load next episode in new window

**Important:** Each episode gets a **fresh, clean Rerun window**:
- ✅ No data overlap between episodes
- ✅ Clean slate for each episode
- ✅ Easy to compare different episodes
- ✅ No confusion from mixed data

**Workflow:**
1. Review episode in Rerun
2. **Close the Rerun window** (X button or Alt+F4)
3. Return to terminal
4. Press Enter for next episode
5. New Rerun window opens with fresh data

## Viewing Modes

### Single Episode Mode
```bash
uv run python view_dataset_local.py data/pick_small_cube_1_20eps 0
```

**Workflow:**
1. Episode 0 loads in fresh Rerun window
2. Review the episode (take as long as you need)
3. **Close the Rerun window** when done
4. Return to terminal and press Enter
5. Type 'y' to load next episode in new window, or 'n' to exit

### Browse All Mode (Recommended)
```bash
uv run python view_dataset_local.py data/pick_small_cube_1_20eps --all
```

**Workflow:**
1. Episode 0 loads in fresh Rerun window
2. Review the episode
3. **Close the Rerun window** when done
4. Return to terminal and press Enter
5. Episode 1 loads in new clean window
6. Repeat until all episodes reviewed

**Controls during browsing:**
- `Enter` - Load next episode
- `s` - Skip to specific episode number
- `q` - Quit browsing mode

### Start from Specific Episode
```bash
# Start browsing from episode 10
uv run python view_dataset_local.py data/pick_small_cube_1_20eps 10 --all
```

## Rerun Viewer Controls

While viewing an episode in Rerun:

### Timeline
- **Scrub bar** - Click/drag to jump to any point in time
- **Play/Pause** - Space bar or play button
- **Speed control** - Adjust playback speed
- **Loop** - Toggle to repeat episode

### View Controls  
- **Pan** - Middle mouse button + drag
- **Zoom** - Scroll wheel
- **Rotate** - Right mouse button + drag (for 3D views)
- **Reset view** - Click reset button

### Data Inspection
- **Click entities** - Select specific data streams
- **Toggle visibility** - Show/hide specific cameras or plots
- **Multi-view** - Arrange multiple views side-by-side

## Quality Checking Workflow

### For 65 Episodes Dataset

**Recommended approach:**

```bash
# Browse all episodes
uv run python view_dataset_local.py data/pick_small_cube_1_20eps --all
```

**For each episode, check:**
1. ✅ **Camera feeds** - Are both cameras working?
2. ✅ **Object visibility** - Can you see the target object clearly?
3. ✅ **Lighting** - Is lighting consistent and adequate?
4. ✅ **Task completion** - Does the robot successfully complete the task?
5. ✅ **Joint positions** - Do motions look smooth (check plots)?
6. ✅ **No anomalies** - Any sudden jumps, occlusions, or issues?

**Keep notes:**
- Good episodes: Just press Enter to continue
- Bad episodes: Note the number (e.g., "Episode 15 - camera occluded")
- Questionable: Review again later using single episode mode

### Reviewing Specific Episodes

If you want to re-examine episode 15:

```bash
uv run python view_dataset_local.py data/pick_small_cube_1_20eps 15
```

## Tips & Tricks

### 1. Fast Review
- Use `--all` mode
- Quickly scrub through timeline
- Press Enter immediately if episode looks good
- Only pause to inspect questionable episodes

### 2. Detailed Review  
- Single episode mode
- Slow down playback in Rerun
- Zoom in on specific moments
- Check joint plots for smoothness

### 3. Comparing Episodes
- Open multiple terminal windows
- Load different episodes in each
- Compare side-by-side

### 4. Taking Screenshots
- Rerun has built-in screenshot button
- Useful for documenting issues
- Can save to file for later reference

## Common Scenarios

### "I want to quickly check all episodes"
```bash
uv run python view_dataset_local.py data/my_dataset --all
# Just press Enter repeatedly, stopping only for issues
```

### "I want to review episodes 10-20"
```bash
uv run python view_dataset_local.py data/my_dataset 10 --all
# Press 'q' when you reach episode 21
```

### "I found a bad episode and want to skip ahead"
```bash
# While in browse mode:
# Press 's'
# Enter: 25
# Continues from episode 25
```

### "I want to keep Rerun open while taking notes"
- ✅ This works! Rerun stays open independently
- Take notes in another window
- Return to terminal when ready for next episode
- Press Enter to continue

## Troubleshooting

### Rerun window not appearing
- Check if it opened on another monitor/desktop
- Look for Rerun in your taskbar
- Try closing any existing Rerun instances first

### Script not waiting for input
- Make sure you're running in interactive terminal
- Don't redirect stdin/stdout
- Use `uv run python`, not background execution

### Want to exit while browsing
- Press 'q' at any prompt
- Or Ctrl+C to force quit
- Rerun viewer will continue running (you can close it manually)

### Rerun viewer is laggy
- Videos are decoded once and kept in memory
- First load might be slower
- Subsequent scrubbing should be smooth
- Reduce window size if needed

## Best Practices

1. ✅ **Use browse mode** for initial quality check
2. ✅ **Keep terminal visible** alongside Rerun viewer
3. ✅ **Take notes** of episode numbers with issues
4. ✅ **Re-review questionable episodes** in single mode
5. ✅ **Don't rush** - quality check is important for good training
6. ✅ **Save good episode numbers** for training subset if needed

