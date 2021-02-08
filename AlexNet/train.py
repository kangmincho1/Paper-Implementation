for epoch in range(2):
  running_loss = 0.0

  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs, labels = data[0].to(device), data[1].to(device)

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
