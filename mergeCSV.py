import sys
import csv

def merge_csv(file_paths, output_path="merged.csv"):
    if not file_paths:
        print("Usage: python mergeCSV.py file1.csv file2.csv [...]")
        sys.exit(1)

    header = None
    rows = []

    for path in file_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            file_header = next(reader)

            if header is None:
                header = file_header
            elif file_header != header:
                print(f"Warning: header mismatch in {path}")

            for row in reader:
                rows.append(row)

    for i, row in enumerate(rows):
        row[0] = str(i)

    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Merged {len(file_paths)} files → {output_path} ({len(rows)} rows)")

if __name__ == "__main__":
    merge_csv(sys.argv[1:])

