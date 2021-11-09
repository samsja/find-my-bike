tests:
	pytest tests

formatting:
	black find_my_bike/
	isort find_my_bike/

notebook-sync:
	jupytext --sync  notebooks/*.ipynb


clean_log:
	rm -rf lightning_logs

tensorboard:
	tensorboard --logdir ./lightning_logs
