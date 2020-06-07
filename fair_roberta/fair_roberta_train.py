def train_model(model_name, X_data, y_data, save_model_name, num_epochs=5):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print("Device: ", device)
	print('Number of training examples: {:,}\n'.format(len(y_data)))

	X_train = torch.tensor([tokenizer.encode(d, pad_to_max_length="True") for d in X_data])
	y_train = torch.tensor(y_data)

	print("X shape:", X_train.shape)
	print("y shape:", y_train.shape)

	# n = tokenizer.encode(X_data[0])
	# n = torch.tensor(n).unsqueeze(0)
	# # print(n)
	# # p = model(n)
	# # print(p[0].data[0][0] > p[0].data[0][1])

	# summary(model, input_data=X_train[0].unsqueeze(0))

	print(model)

	batch_size = 64
	dataset = Dataset(X_train, y_train)
	loader = DataLoader(dataset, batch_size, shuffle=True)
	model = model.to(device)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters())

	for epoch in range(num_epochs):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(loader, 0):
		# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()
			# forward + backward + optimize
			outputs = model(inputs)

			loss = criterion(outputs[0], labels)
			loss.backward()
			optimizer.step()
			# print statistics

			running_loss += loss.item()

			if i % 20 == 19:    # print every 20 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/20))
				running_loss = 0.0

		torch.save(model, save_model_name)