.PHONY: install test run docker-build clean

install:
	pip install -r requirements.txt

test:
	export PYTHONPATH=$$PYTHONPATH:. && pytest tests/

run:
	python main_pipeline.py --duration 4000 --output VEP_Report.html

docker-build:
	docker-compose build

docker-run:
	docker-compose up

clean:
	rm -f *.html *.npz
	rm -rf __pycache__
