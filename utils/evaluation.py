from .helpers import test

def evaluate_model(model, test_loader, criterion, device, classes):
    test_loss, test_acc = test(model, test_loader, criterion, device, classes)
    return test_loss, test_acc
