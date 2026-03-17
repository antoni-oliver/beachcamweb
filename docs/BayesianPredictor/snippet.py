import os
from datetime import timedelta
from predictions.classes.BayesianPredictor import BayesianPredictor

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # To download pytorch model

predictors = [BayesianPredictor()]
IMG_PATH = "img.jpg"
MASK_PATH = "mask.jpg"

def main():
	try:
		if snapshot.mask_sand_and_water:
			mask_paths = [MASK_PATH]
		else:
			mask_paths = []

		predictionDTO = predictor.predict(IMG_PATH, mask_paths)
		print(f"  Prediction done: predictionDTO.crowd_count)
		prediction_image_path = beachcam.relative_filepath(timestamp=snapshot.ts, subfolder='img/predictions/', extension='.jpg')
		
		with open("./prediction_output.jpg"), 'wb') as f:
			f.write(predictionDTO.img_predict_content)
		print('  Prediction saved.')
	except Exception as e:
		# Handle any exception
		print(f"An error ocurred: {e}")


if __name__ == "__main__":
    main()
