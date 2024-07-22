OUT_FILE = out.csv

all: run

run: $(OUT_FILE)
	python check_values.py $(OUT_FILE)

$(OUT_FILE): main.py
	python main.py > $(OUT_FILE)

clean:
	rm -f $(OUT_FILE)

.PHONY: all run clean
