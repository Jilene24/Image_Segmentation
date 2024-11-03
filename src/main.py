import torch
from torch.utils.data import DataLoader
from environment_setup import EnvironmentSetup
from data_preparation import DataPreparation
from augmentations import Augmentations
from segmentation_dataset import SegmentationDataset
from segmentation_model import SegmentationModel
from training import Training
from visualizer import Visualizer

# Main execution
if __name__ == "__main__":
    # Step 1: Environment setup
    EnvironmentSetup.install_dependencies()
    EnvironmentSetup.clone_repository("https://github.com/parth1620/Human-Segmentation-Dataset-master.git")

    # Step 2: Data preparation
    data_preparation = DataPreparation(
        csv_file='/content/Human-Segmentation-Dataset-master/train.csv',
        data_dir='/content/'
    )
    train_df, valid_df = data_preparation.split_data()

    # Step 3: Create datasets and dataloaders
    trainset = SegmentationDataset(train_df, Augmentations.get_train_augs())
    validset = SegmentationDataset(valid_df, Augmentations.get_valid_augs())
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=16)

    # Step 4: Initialize model
    model = SegmentationModel()
    model.to('cuda')

    # Step 5: Train the model
    trainer = Training(model, trainloader, valid_loader)
    trainer.run_training()

    # Step 6: Visualize results
    idx = 20
    model.load_state_dict(torch.load('best_model.pt'))
    image, mask = validset[idx]
    logits_mask = model(image.to('cuda').unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask > 0.5) * 1.0

    Visualizer.show_image(image, mask)
    Visualizer.show_image(image, pred_mask.detach().cpu().squeeze(0))
