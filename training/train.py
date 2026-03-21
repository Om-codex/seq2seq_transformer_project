def train(model, num_epochs, dataloader, optimizer, criterion):
    
    for epoch in range(num_epochs):
        total_loss = 0
        for src, trg in dataloader:
            optimizer.zero_grad()

            output = model(src, trg) # output: [batch_size, trg_len, vocab_size]

            output = output.view(-1, output.shape[-1])
            target = trg.view(-1)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/ len(dataloader)
        print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

