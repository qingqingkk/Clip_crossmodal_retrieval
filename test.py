import parser
import torch
from util import encode_dataset, get_map

def metrics_at_k(image_input, text_input, model, Loader, k_vals, batch_size, train_mode):
    device = model.device
    print("Encoding all data...")
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    
    if train_mode == 'total':
        image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(model, Loader, batch_size=batch_size)
    else:
        image_encodings, text_encodings = model(image_input, text_input)
        text_to_image_map, image_to_text_map = get_map(Loader, batch_size)

    image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
    text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)


    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    
    #（cos similarity）
    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    
    #print("Text-to-image recall...")
    
    text_to_image_recall = []

    for k in k_vals:
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1).cpu()

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    #print("Text-to-image mAP...")
    
    mAP_t2i=[]
    
    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]
        precision_calculator = 0
        #num_correct_calculator = 0

        for i in range(k):
            #print(i)
            the_ith_retrieval = topk[:, i] # chang into 1D

            # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
            correct = torch.eq(the_ith_retrieval, text_to_image_map).cpu()
            num_correct = torch.sum(correct).item()
            #print(num_correct)
            #num_correct_calculator += num_correct
            
            precision = num_correct / (i+1)
            precision_calculator += precision
        AP_sum = precision_calculator / 1
        mAP_t2i.append(AP_sum / num_text)


    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    
    #print("Image-to-text recall...")

    image_to_text_recall = []

    for k in k_vals:
        
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).to(device)

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#
        

    #print("Image-to-text mAP...")

    mAP_i2t=[]
    for k in k_vals:
        
        topk = inds[:, :k]
        precision_calculator = 0
        #num_correct_calculator = 0

        AP = []

        for im in range(num_im):
            
            topk_indices = topk[im]  # K describtion
            relevant_indices = image_to_text_map[im]  # 5 relevant captions for this image

            num_relevant_items_found = 0
            precision_at_i = 0
            sum_precisions = 0
            num_precisions = 0

            for rank, prediction in enumerate(topk_indices):
                if prediction in relevant_indices:
                    num_relevant_items_found += 1
                    precision_at_i = num_relevant_items_found / (rank + 1)
                    sum_precisions += precision_at_i
                    num_precisions += 1

            average_precision = sum_precisions / num_precisions if num_precisions else 0
            
            AP.append(average_precision)

        mAP_i2t.append(sum(AP) / len(AP))

    return text_to_image_recall, image_to_text_recall, mAP_t2i, mAP_i2t


def test(image_test, text_test, model, test_loader, train_mode):

    recall_t2i, recall_i2t, mAP_t2i, mAP_i2t = metrics_at_k(image_test, text_test, model, Loader = test_loader, k_vals=[1,5,10], batch_size=test_loader.batch_size, train_mode=train_mode)

    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, recall_t2i):
        print(f" R@{k}: {100*x:.2f}%")

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, recall_i2t):
        print(f" R@{k}: {100*x:.2f}%")

    print("Text-to-image mAP@K")
    for k, x in zip(k_vals, mAP_t2i):
        print(f" mAP@{k}: {100*x:.2f}%")

    print("Image-to-text mAP@K")
    for k, x in zip(k_vals, mAP_i2t):
        print(f" mAP@{k}: {100*x:.2f}%")


