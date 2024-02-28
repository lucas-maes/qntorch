import numpy as np
import torch

class OptimTracker:

	def __init__(self):
		self.data = {}
		return

	def track(self, **kwargs):

		for key, value in kwargs.items():
			self._track_item(key, value)

		return

	def _track_item(self, key, value):
		
		if (type(value) == np.ndarray) or (type(value) == torch.Tensor):
			value = value.tolist()
			if type(value) == list:
				if len(value) == 1:
					value = value[0]

		if key not in self.data:
			self.data[key] = [value]

		else:
			self.data[key].append(value)

		return


	def get(self, key, numpy=False, torch=False):

		data = self.data[key] if key in self.data else None

		if data:
			if torch:
				data = torch.tensor(data)
			if numpy:
				data = np.array(data)

		return data

	def remove(self, key):
		if key in self.data:
			del self.data[key]

	def plot(self, key):
		pass

	def print(self, key):

		if key not in self.data:
			print(f"{key} key not stored in tracker.")
			return

		data = self.data[key]

		for item in data:
			print(item)

	def __call__(self, **kwargs):
		self.track(**kwargs)
