import os
import glob
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

def main():
    log_dir = input("Enter the path to your TensorBoard log directory: ").strip()
    if not os.path.isdir(log_dir):
        print("Provided path is not a valid directory.")
        return

    try:
        step_threshold = int(input("Enter the maximum step to KEEP (e.g. 26257): ").strip())
    except ValueError:
        print("Invalid number for step.")
        return

    log_dir_filtered = log_dir.rstrip("/\\") + "_filtered"
    os.makedirs(log_dir_filtered, exist_ok=True)

    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not event_files:
        print("No event files found in the directory.")
        return

    writer = SummaryWriter(log_dir_filtered)

    max_step_written = 0

    for file in event_files:
        ea = event_accumulator.EventAccumulator(file)
        try:
            ea.Reload()
        except Exception as e:
            print(f"Skipping corrupted or unreadable file: {file}\n{e}")
            continue

        for tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            for ev in events:
                if ev.step <= step_threshold:
                    writer.add_scalar(tag, ev.value, ev.step)
                    max_step_written = max(max_step_written, ev.step)


    writer.close()

    print(f"\nFiltered logs saved to: {log_dir_filtered}")
    print(f"Maximum step retained in filtered logs: {max_step_written}")

if __name__ == "__main__":
    main()