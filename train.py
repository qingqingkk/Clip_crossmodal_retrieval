import torch
from util import get_dataset, train_setup, plot_loss
from test import test

def val(val_loader, model, loss, train_mode):
    model.eval()
    tot_loss = 0
    for image, text in val_loader:
        random_indices = torch.randint(0, 5, (len(image),))
        text = torch.stack([text[i, idx] for i, idx in enumerate(random_indices)]).to(device)
        image = image.to(device)
       
        if train_mode == 'total':
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
        else:
            image_features, text_features = model(image, text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        targets = torch.arange(len(image),dtype=torch.long, device=device)

        val_loss = loss.forward(image_features, text_features, targets)

        tot_loss += val_loss.cpu()
    return tot_loss/len(val_loader)

def Fine_Tune(args):

    model, optimizer, lr_scheduler, loss_func = train_setup(args)
    train_loader, val_loader = get_dataset(args)
    image_test, text_test, test_loader = test_dataset(args)
    model_name = args.dataset + '_' + args.train_mode + '_' + args.model_version
    model_dir = args.model_path
    train_mode = args.train_mode
    print(f'We will fine tune the {train_mode} structure')
    min_loss = np.inf
    for epoch in range(args.max_epochs):
        batch_num = 0
        train_loss_list = []
        val_loss_list = []
        for image, text in tqdm(train_loader):
            random_indices = torch.randint(0, 5, (len(image),))
            text = torch.stack([text[i, idx] for i, idx in enumerate(random_indices)]).to(device)
            image = image.to(device)
            batch_num += 1

            optimizer.zero_grad()
            if train_mode == 'total':
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
            else:
                image_features, text_features = model(image, text)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            targets = torch.arange(len(image),dtype=torch.long, device=device)
            
            train_loss = loss_func.forward(image_features, text_features, targets)
            tot_loss += train_loss.cpu()
            train_loss.backward()
            optimizer.step()

        cur_loss = val(val_loader, model, loss_func)
        train_loss_list.append(tot_loss.item())
        val_loss_list.append(cur_loss.item())

        if cur_loss < min_loss:
            min_loss = cur_lossF
            torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}.pth"))
            print(f"min val loss: {min_loss}")
        
        lr_scheduler.step(cur_loss)
        test(image_test, text_test, model, test_loader = test_loader, train_mode = train_mode)

    plot_loss(train_loss_list, val_loss_list, args)


def Zero_Shot(args):
    train_mode = args.train_mode
    model = train_setup(args)
    image_test, text_test, test_loader = test_dataset(args)
    test(image_test, text_test, model, test_loader = test_loader, train_mode = train_mode)



    