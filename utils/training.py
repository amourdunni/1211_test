import torch
import os
from .helpers import train_one_epoch, validate, plot_metrics
import logging

def fine_tuning(model, train_loader, val_loader, optimizer, criterion, scheduler, device, scaler, config):
    best_val_acc = 0.0
    best_model_state = model.state_dict()
    trigger_times = 0
    patience = config['training']['patience']
    save_path = config['training']['save_path']
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        logging.info(f"Epoch [{epoch+1}/{config['training']['num_epochs']}]")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            logging.info(f"--> Best model saved with Val Acc: {best_val_acc:.2f}%")
            trigger_times = 0
        else:
            trigger_times += 1
            logging.info(f"검증 정확도가 {trigger_times} 에포크 동안 개선되지 않았습니다.")
        
        if trigger_times >= patience:
            logging.info("조기 종료가 트리거되었습니다.")
            break
    
    model.load_state_dict(best_model_state)
    logging.info(f"최고 검증 정확도: {best_val_acc:.2f}%")

    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
    
    return model
