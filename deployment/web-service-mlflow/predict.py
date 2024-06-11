import pickle
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = '1f576c2ab4cb46e7a3d4d6578b006ed4' # Add the run ID of the model
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
	features = {
		'PU_DO': f"{ride['PULocationID']}_{ride['DOLocationID']}",
		'trip_distance': ride['trip_distance']
	}

	return features

def predict(features):
	preds = model.predict(features)[0]
	return preds

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
	ride = request.get_json()

	features = prepare_features(ride)
	pred = predict(features)

	result = {
		'duration': pred,
		'model_version': RUN_ID
	}

	return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=9696)