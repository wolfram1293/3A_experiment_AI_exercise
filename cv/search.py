
import argparse
import torch
from torchvision import transforms
from PIL import Image
from network_db import Vgg16
# 4.3.4で変更済み
def search(src, db_features, db_paths, k, gpu):
	from network import MLP
	model = MLP(1000, 28*28, 10)
	model.load_state_dict(torch.load('result/model_final'))
	# Set transformation
	data_preprocess = transforms.Compose([
		transforms.ToTensor(),])
	# Set model to GPU/CPU
	device = 'cpu'
	if gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(gpu)
	model = model.to(device)
	# Get features
	with torch.no_grad():
		src_feature = model.fc2(model.fc1(data_preprocess(Image.open(src, 'r')).flatten().unsqueeze(0).to(device))).to('cpu')
	# Load database
	print(src_feature.shape)
	paths = torch.load(db_paths)
	features = torch.load(db_features)
	assert k <= len(paths)
	assert len(features) == len(paths)
	# Calculate distances
	distances = torch.tensor(
		[torch.norm(src_feature - feature)
			for feature in features]
	)
	_, indices = torch.topk(distances, k, largest=False)
	# Show results
	for i in indices:
		print(paths[i])

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(search image)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--input', '-i', default='default_source_image_path',
						help='path to database features')
	parser.add_argument('--features', '-f', default='result/feature.pt',
						help='path to database features')
	parser.add_argument('--paths', '-p', default='result/path.pt',
						help='path to database paths')
	parser.add_argument('--k', '-k', type=int, default=5,
						help='find num')
	args = parser.parse_args()

	search(args.input, args.features, args.paths, args.k, args.gpu)

if __name__ == '__main__':
	main()