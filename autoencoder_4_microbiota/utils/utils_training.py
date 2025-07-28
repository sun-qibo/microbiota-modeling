
def get_model_filename(model_name, mask_x_true, mask_type, num_layers_to_freeze, latent_dim, dropout, num_epochs_1, patience1, num_epochs_2, patience2, alpha):
    return f"{model_name.lower()}_maskxtrue_{str(mask_x_true).lower()}_masktype_{mask_type}_numlayerstofreeze_{num_layers_to_freeze}_alpha_{int(alpha*100)}_latentdim_{latent_dim}_dropout_{int(dropout * 100)}_{num_epochs_1}_{patience1}_{num_epochs_2}_{patience2}"


def early_stopping(epoch, val_loss, min_val_loss, best_model, model, early_stopping_counter, patience):
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_model = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping")
            return True, min_val_loss, best_model, early_stopping_counter
    return False, min_val_loss, best_model, early_stopping_counter