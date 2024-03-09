class Algorithm:

	def __init__(self, x0, f, tracker,
				n_iter=100,
				use_eps=True, 
				eps=10e-10
		):

		# optim tracker
		# -------------

		self.tracker = tracker      # metrics tracker

		# solver params
		# -------------

		self.n_iter = n_iter 		# max number of iterations
		self.use_eps = use_eps 		# use norm between two last iterates difference as stopping criterion
		self.eps = eps 				# precision of two last iterates difference norm to stop
		self.x0 = x0				# initial point
		self.f = f					# function to optimize
		self.size = x0.size(0)		# size of the problem

		return

	def step(self, x):
		raise NotImplementedError

	def solve(self):
		"""
		Find optimum of f(x) starting from x0 using self.step()

		returns:
		--------
		x: the optimum find given the algorithm params (tensor)
		"""

		x = self.x0				# current iterate
		x_last = None		# last iterate

		for step in range(self.n_iter):

			# current iterate become last iterate
			x_last = x

			# take a optimization step define our new point
			x = self.step(x)

			if self.use_eps:
				# compute norm between diff of two last iterates
				iter_dist = (x-x_last).norm(p=2)

				# log
				self.tracker(x_last=x_last,
							x=x,
							step=step+1,
							iter_dist=iter_dist.item())

				# if small enough stop training
				if iter_dist <= self.eps:
					break

		return x

	def __call__(self, x0):
		return self.solve(x0)