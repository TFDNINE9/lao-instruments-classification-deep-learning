import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_json_data(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_history(data, save_path='training_history_plots.png'):
    """
    Create comprehensive training history plots
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training Loss for all folds
    ax1 = plt.subplot(3, 3, 1)
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        epochs = range(1, len(history['loss']) + 1)
        
        ax1.plot(epochs, history['loss'], label=f'Fold {fold_num}', 
                linewidth=2, alpha=0.8)
    
    ax1.set_title('Training Loss - All Folds', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # 2. Validation Loss for all folds
    ax2 = plt.subplot(3, 3, 2)
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        epochs = range(1, len(history['val_loss']) + 1)
        
        ax2.plot(epochs, history['val_loss'], label=f'Fold {fold_num}', 
                linewidth=2, alpha=0.8, linestyle='--')
    
    ax2.set_title('Validation Loss - All Folds', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Training vs Validation Loss (Average)
    ax3 = plt.subplot(3, 3, 3)
    
    # Calculate average losses
    max_epochs = max(len(f['history']['loss']) for f in data)
    avg_train_loss = np.zeros(max_epochs)
    avg_val_loss = np.zeros(max_epochs)
    count = np.zeros(max_epochs)
    
    for fold_data in data:
        history = fold_data['history']
        n_epochs = len(history['loss'])
        avg_train_loss[:n_epochs] += history['loss']
        avg_val_loss[:n_epochs] += history['val_loss']
        count[:n_epochs] += 1
    
    count[count == 0] = 1
    avg_train_loss /= count
    avg_val_loss /= count
    
    # Only plot up to where we have data
    valid_epochs = np.where(count > 0)[0]
    if len(valid_epochs) > 0:
        max_valid_epoch = valid_epochs[-1] + 1
        epochs_plot = range(1, max_valid_epoch + 1)
        ax3.plot(epochs_plot, avg_train_loss[:max_valid_epoch], 'b-', label='Avg Train Loss', linewidth=3)
        ax3.plot(epochs_plot, avg_val_loss[:max_valid_epoch], 'r-', label='Avg Val Loss', linewidth=3)
        ax3.fill_between(epochs_plot, avg_train_loss[:max_valid_epoch], avg_val_loss[:max_valid_epoch], alpha=0.3, color='gray')
    
    ax3.set_title('Average Loss Across Folds', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend(loc='upper right', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Training Accuracy for all folds
    ax4 = plt.subplot(3, 3, 4)
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        epochs = range(1, len(history['accuracy']) + 1)
        
        ax4.plot(epochs, history['accuracy'], label=f'Fold {fold_num}', 
                linewidth=2, alpha=0.8)
    
    ax4.set_title('Training Accuracy - All Folds', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    # 5. Validation Accuracy for all folds
    ax5 = plt.subplot(3, 3, 5)
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        epochs = range(1, len(history['val_accuracy']) + 1)
        
        ax5.plot(epochs, history['val_accuracy'], label=f'Fold {fold_num}', 
                linewidth=2, alpha=0.8, linestyle='--')
    
    ax5.set_title('Validation Accuracy - All Folds', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.legend(loc='lower right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.05])
    
    # 6. Training vs Validation Accuracy (Average)
    ax6 = plt.subplot(3, 3, 6)
    
    # Calculate average accuracies
    avg_train_acc = np.zeros(max_epochs)
    avg_val_acc = np.zeros(max_epochs)
    count = np.zeros(max_epochs)
    
    for fold_data in data:
        history = fold_data['history']
        n_epochs = len(history['accuracy'])
        avg_train_acc[:n_epochs] += history['accuracy']
        avg_val_acc[:n_epochs] += history['val_accuracy']
        count[:n_epochs] += 1
    
    count[count == 0] = 1
    avg_train_acc /= count
    avg_val_acc /= count
    
    # Only plot up to where we have data
    valid_epochs = np.where(count > 0)[0]
    if len(valid_epochs) > 0:
        max_valid_epoch = valid_epochs[-1] + 1
        epochs_plot = range(1, max_valid_epoch + 1)
        ax6.plot(epochs_plot, avg_train_acc[:max_valid_epoch], 'b-', label='Avg Train Acc', linewidth=3)
        ax6.plot(epochs_plot, avg_val_acc[:max_valid_epoch], 'r-', label='Avg Val Acc', linewidth=3)
        ax6.fill_between(epochs_plot, avg_train_acc[:max_valid_epoch], avg_val_acc[:max_valid_epoch], alpha=0.3, color='gray')
    
    ax6.set_title('Average Accuracy Across Folds', fontsize=16, fontweight='bold')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.legend(loc='lower right', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.05])
    
    # 7. Learning Rate Schedule
    ax7 = plt.subplot(3, 3, 7)
    
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        if 'lr' in history:
            epochs = range(1, len(history['lr']) + 1)
            ax7.plot(epochs, history['lr'], label=f'Fold {fold_num}', 
                    linewidth=2, alpha=0.8)
    
    ax7.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Learning Rate', fontsize=12)
    ax7.set_yscale('log')
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Final Metrics Comparison
    ax8 = plt.subplot(3, 3, 8)
    
    fold_nums = [f['fold'] for f in data]
    val_accs = [f['val_accuracy'] for f in data]
    val_losses = [f['val_loss'] for f in data]
    
    x = np.arange(len(fold_nums))
    width = 0.35
    
    ax8_twin = ax8.twinx()
    
    bars1 = ax8.bar(x - width/2, val_accs, width, label='Val Accuracy', 
                    color='skyblue', edgecolor='navy', linewidth=2)
    bars2 = ax8_twin.bar(x + width/2, val_losses, width, label='Val Loss', 
                        color='lightcoral', edgecolor='darkred', linewidth=2)
    
    # Add value labels
    for bar, acc in zip(bars1, val_accs):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar, loss in zip(bars2, val_losses):
        height = bar.get_height()
        ax8_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{loss:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax8.set_xlabel('Fold', fontsize=12)
    ax8.set_ylabel('Validation Accuracy', fontsize=12, color='navy')
    ax8_twin.set_ylabel('Validation Loss', fontsize=12, color='darkred')
    ax8.set_xticks(x)
    ax8.set_xticklabels([f'Fold {n}' for n in fold_nums])
    ax8.tick_params(axis='y', labelcolor='navy')
    ax8_twin.tick_params(axis='y', labelcolor='darkred')
    ax8.set_ylim([0.95, 1.0])
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Final Validation Metrics', fontsize=16, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax8.get_legend_handles_labels()
    lines2, labels2 = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines1 + lines2, labels1 + labels2, loc='lower center')
    
    # 9. Training Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    avg_final_val_acc = np.mean(val_accs)
    std_val_acc = np.std(val_accs)
    best_fold = fold_nums[np.argmax(val_accs)]
    best_acc = max(val_accs)
    avg_epochs = np.mean([len(f['history']['loss']) for f in data])
    
    # Calculate overfitting metrics
    overfitting_gaps = []
    for fold_data in data:
        history = fold_data['history']
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        gap = final_train_acc - final_val_acc
        overfitting_gaps.append(gap)
    
    avg_overfit_gap = np.mean(overfitting_gaps)
    
    # Display summary
    summary_text = f"""Training Summary
    
Average Val Accuracy: {avg_final_val_acc:.4f} ± {std_val_acc:.4f}
Best Fold: {best_fold} (Accuracy: {best_acc:.4f})
Average Epochs: {avg_epochs:.1f}
Average Overfit Gap: {avg_overfit_gap:.4f}

Per-Fold Results:
"""
    
    for i, (fold, acc, loss, gap) in enumerate(zip(fold_nums, val_accs, val_losses, overfitting_gaps)):
        summary_text += f"\nFold {fold}: Acc={acc:.4f}, Loss={loss:.4f}, Gap={gap:.4f}"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Enhanced Model Training History Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'avg_final_val_acc': avg_final_val_acc,
        'std_val_acc': std_val_acc,
        'best_fold': best_fold,
        'best_accuracy': best_acc,
        'avg_epochs': avg_epochs,
        'avg_overfit_gap': avg_overfit_gap
    }

def plot_convergence_analysis(data, save_path='convergence_analysis.png'):
    """
    Create convergence analysis plots
    """
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Best Epoch Analysis
    ax1 = plt.subplot(1, 3, 1)
    
    best_epochs = []
    fold_nums = []
    
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        best_epoch = np.argmax(history['val_accuracy']) + 1
        best_epochs.append(best_epoch)
        fold_nums.append(fold_num)
    
    ax1.scatter(fold_nums, best_epochs, s=200, c='purple', alpha=0.7, 
               edgecolors='black', linewidth=2)
    ax1.plot(fold_nums, best_epochs, 'purple', alpha=0.3, linewidth=2)
    
    avg_best_epoch = np.mean(best_epochs)
    ax1.axhline(y=avg_best_epoch, color='red', linestyle='--', 
               label=f'Average: {avg_best_epoch:.1f}', linewidth=2)
    
    ax1.set_title('Best Epoch per Fold', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Best Epoch', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence Speed
    ax2 = plt.subplot(1, 3, 2)
    
    convergence_epochs = []
    
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        val_acc = history['val_accuracy']
        
        # Find epoch where accuracy reaches 95% of final value
        final_acc = val_acc[-1]
        target_acc = 0.95 * final_acc
        
        convergence_epoch = next((i+1 for i, acc in enumerate(val_acc) 
                                 if acc >= target_acc), len(val_acc))
        convergence_epochs.append(convergence_epoch)
    
    bars = ax2.bar(fold_nums, convergence_epochs, color='lightgreen', 
                   edgecolor='darkgreen', linewidth=2)
    
    for bar, epoch in zip(bars, convergence_epochs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                str(epoch), ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Convergence Speed (95% of Final Accuracy)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Epochs to Converge', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Stability Analysis
    ax3 = plt.subplot(1, 3, 3)
    
    for fold_data in data:
        fold_num = fold_data['fold']
        history = fold_data['history']
        
        # Calculate validation accuracy stability (rolling std)
        val_acc = history['val_accuracy']
        window_size = min(10, len(val_acc) // 4)
        
        if len(val_acc) > window_size:
            rolling_std = []
            for i in range(window_size, len(val_acc)):
                window = val_acc[i-window_size:i]
                rolling_std.append(np.std(window))
            
            epochs = range(window_size, len(val_acc))
            ax3.plot(epochs, rolling_std, label=f'Fold {fold_num}', 
                    linewidth=2, alpha=0.8)
    
    ax3.set_title('Validation Accuracy Stability (Rolling Std)', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Rolling Std Dev', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Model Convergence Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training history from JSON file')
    parser.add_argument('json_file', type=str, help='Path to JSON file containing training history')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for plots')
    parser.add_argument('--show_convergence', action='store_true', help='Also plot convergence analysis')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.json_file}...")
    data = load_json_data(args.json_file)
    
    # Create main plots
    print("Creating training history plots...")
    main_plot_path = output_dir / 'training_history_plots.png'
    summary = plot_training_history(data, save_path=str(main_plot_path))
    
    print("\nTraining Summary:")
    print(f"Average Validation Accuracy: {summary['avg_final_val_acc']:.4f} ± {summary['std_val_acc']:.4f}")
    print(f"Best Fold: {summary['best_fold']} (Accuracy: {summary['best_accuracy']:.4f})")
    print(f"Average Epochs: {summary['avg_epochs']:.1f}")
    print(f"Average Overfitting Gap: {summary['avg_overfit_gap']:.4f}")
    
    # Create convergence analysis if requested
    if args.show_convergence:
        print("\nCreating convergence analysis plots...")
        conv_plot_path = output_dir / 'convergence_analysis.png'
        plot_convergence_analysis(data, save_path=str(conv_plot_path))
    
    print(f"\nPlots saved to {output_dir}")

if __name__ == "__main__":
    # If running without command line arguments, you can use it like this:
    # data = load_json_data('cv_results.json')
    # plot_training_history(data)
    # plot_convergence_analysis(data)
    
    main()