
import argparse
import glob
import os
import torch
from torchvision import transforms
from PIL import Image
from network_db import Vgg16

def createDatabase(paths, gpu):
	# Create model
  from network import MLP
  model = MLP(1000, 28*28, 10)
  model.load_state_dict(torch.load('result/model_final'))
  # Set transformation
  data_preprocess = transforms.Compose([transforms.ToTensor()])
  # Set model to GPU/CPU
  device = 'cpu'
  if gpu >= 0:
      # Make a specified GPU current
      device = 'cuda:' + str(gpu)
  model = model.to(device)
  # Get features
  with torch.no_grad():
      features = torch.cat(
      [model.fc2(model.fc1(data_preprocess(Image.open(path, 'r')).flatten().unsqueeze(0).to(device))).to('cpu')
              for path in paths],
          dim = 0
      )
  # Show created dataset size
  print('dataset size : {}'.format(len(features)))
  return features

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(create database)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--dataset', '-d', default='default_dataset_path',
						help='Directory for creating database')
	args = parser.parse_args()

	data_dir = args.dataset


	# Get a list of pictures
	paths = glob.glob(os.path.join(data_dir, './*/*.png'))

	assert len(paths) != 0 
	# Create the database
	features = createDatabase(paths, args.gpu)
	# Save the data of database
	torch.save(features, 'result/feature.pt')
	torch.save(paths, 'result/path.pt')

if __name__ == '__main__':
	main()
