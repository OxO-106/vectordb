import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import curve_fit
import os
import json
from scipy import stats

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Functions from reduce.py
def parse_embedding(embedding_str: str) -> np.ndarray:
    """Convert string representation of embedding to numpy array"""
    values = embedding_str.strip('[]').split(',')
    return np.array([float(x) for x in values])

def apply_pca(embeddings: np.ndarray, n_components: int) -> tuple:
    """
    Apply PCA to reduce dimensionality of embeddings.

    Args:
        embeddings: numpy array of embeddings
        n_components: target number of dimensions

    Returns:
        tuple: (reduced embeddings, PCA model, explained variance ratio)
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)

    return reduced_embeddings, pca, explained_variance

# Functions from precision.py
def parse_vector(vec_str):
    """Parse vector string into numpy array"""
    if isinstance(vec_str, str):
        try:
            return np.array(json.loads(vec_str))
        except:
            return None
    else:
        return vec_str  # If it's already a list or array

def combine_embeddings(image_vectors, caption_vectors):
    """Concatenate image and caption embeddings"""
    return np.concatenate((image_vectors, caption_vectors), axis=1)

def precision_at_k(original_neighbors, reduced_neighbors, k):
    """Calculate Precision@K by comparing neighbor sets"""
    original_set = set(original_neighbors[:k])
    reduced_set = set(reduced_neighbors[:k])
    intersection = original_set.intersection(reduced_set)
    return len(intersection) / k

# Curve fitting functions
def simple_logistic(x, L, k, x0):
    """Safer logistic function with clipping to prevent overflow"""
    return L / (1 + np.exp(-k * np.clip((x - x0), -100, 100)))

def simple_exp(x, a, b, c):
    """Simple exponential approach function with clipping"""
    return a * (1 - np.exp(-b * np.clip(x, 0, 700))) + c

def simple_log(x, a, b):
    """Simple logarithmic function"""
    return a * np.log(x + 1) + b

def calculate_derivative(x, params, func):
    """Calculate the numerical derivative of the fitted function at each point"""
    h = 0.01  # Small step for numerical differentiation
    return [(func(x_i + h, *params) - func(x_i - h, *params)) / (2 * h) for x_i in x]

def process_original_data(input_file='data/embedding.csv'):
    """Process original embeddings and create reduced dimension versions"""
    print("Loading original embeddings...")
    df = pd.read_csv(input_file)
    
    # Convert embeddings to numpy arrays
    print("Processing image embeddings...")
    image_embeddings = np.vstack(df['image_embedding'].apply(parse_embedding))
    print("Processing caption embeddings...")
    caption_embeddings = np.vstack(df['caption_embedding'].apply(parse_embedding))
    
    print(f"Original embedding shapes: Image {image_embeddings.shape}, Caption {caption_embeddings.shape}")
    
    return df, image_embeddings, caption_embeddings

def generate_reduced_embeddings(image_embeddings, caption_embeddings, dims):
    """Generate reduced embeddings for specified dimensions"""
    results = {}
    
    for dim in dims:
        print(f"Reducing to {dim} dimensions...")
        
        # Apply PCA to both types of embeddings
        image_reduced, _, image_variance = apply_pca(image_embeddings, dim)
        caption_reduced, _, caption_variance = apply_pca(caption_embeddings, dim)
        
        print(f"  Image variance: {image_variance:.4f}, Caption variance: {caption_variance:.4f}")
        
        # Store the reduced embeddings
        results[dim] = {
            'image_embeddings': image_reduced,
            'caption_embeddings': caption_reduced,
            'image_variance': image_variance,
            'caption_variance': caption_variance
        }
    
    return results

def calculate_precision_metrics(original_image_vecs, original_caption_vecs, reduced_embeddings, k_values=[5, 10, 15]):
    """Calculate precision metrics for all reduced embeddings"""
    # Calculate original combined embeddings and distances
    original_combined = combine_embeddings(original_image_vecs, original_caption_vecs)
    original_distances = euclidean_distances(original_combined)
    
    # Get original nearest neighbors (excluding self)
    n_samples = len(original_combined)
    max_k = max(k_values)
    original_neighbors = np.argsort(original_distances, axis=1)[:, 1:max_k+1]
    
    # Calculate precision for each reduced dimension
    precision_results = {}
    
    for dim, embeddings in reduced_embeddings.items():
        reduced_caption_vecs = embeddings['caption_embeddings']
        
        # Combine with original image embeddings (as per precision.py)
        reduced_combined = combine_embeddings(original_image_vecs, reduced_caption_vecs)
        reduced_distances = euclidean_distances(reduced_combined)
        reduced_neighbors = np.argsort(reduced_distances, axis=1)[:, 1:max_k+1]
        
        # Calculate Precision@K for each k
        dim_results = {}
        for k in k_values:
            precisions = []
            for i in range(n_samples):
                p_at_k = precision_at_k(original_neighbors[i], reduced_neighbors[i], k)
                precisions.append(p_at_k)
                
            dim_results[k] = {
                'mean': np.mean(precisions),
                'std': np.std(precisions)
            }
            
        precision_results[dim] = dim_results
    
    return precision_results

def fit_curves(dimensions, precision_values):
    """Fit various curves to precision data and find optimal range"""
    x_data = np.array(dimensions)
    y_data = np.array(precision_values)
    
    # Define x points for smooth curves
    x_smooth = np.linspace(min(x_data), max(x_data), 1000)
    
    # Dictionary to store curve data
    curve_results = {}
    derivatives = {}
    
    # Try logistic function (often good for precision curves)
    try:
        logistic_params, _ = curve_fit(simple_logistic, x_data, y_data, 
                                     p0=[1.0, 0.01, 50], maxfev=5000)
        logistic_y = simple_logistic(x_smooth, *logistic_params)
        logistic_rmse = np.sqrt(np.mean((simple_logistic(x_data, *logistic_params) - y_data)**2))
        curve_results['Logistic'] = {
            'rmse': logistic_rmse,
            'params': logistic_params,
            'func': simple_logistic,
            'y': logistic_y
        }
        
        # Calculate numerical derivative
        try:
            derivatives['Logistic'] = calculate_derivative(x_smooth, logistic_params, simple_logistic)
        except Exception as e:
            print(f"Warning: Could not calculate logistic derivative: {e}")
            derivatives['Logistic'] = np.zeros_like(x_smooth)
    except Exception as e:
        print(f"Warning: Could not fit logistic function: {e}")
    
    # Try log function
    try:
        log_params, _ = curve_fit(simple_log, x_data, y_data)
        log_y = simple_log(x_smooth, *log_params)
        log_rmse = np.sqrt(np.mean((simple_log(x_data, *log_params) - y_data)**2))
        curve_results['Log'] = {
            'rmse': log_rmse,
            'params': log_params,
            'func': simple_log,
            'y': log_y
        }
        
        # Calculate numerical derivative
        try:
            derivatives['Log'] = calculate_derivative(x_smooth, log_params, simple_log)
        except Exception as e:
            print(f"Warning: Could not calculate log derivative: {e}")
            derivatives['Log'] = np.zeros_like(x_smooth)
    except Exception as e:
        print(f"Warning: Could not fit log function: {e}")
    
    # Try exponential approach
    try:
        exp_params, _ = curve_fit(simple_exp, x_data, y_data, p0=[0.3, 0.01, 0.7])
        exp_y = simple_exp(x_smooth, *exp_params)
        exp_rmse = np.sqrt(np.mean((simple_exp(x_data, *exp_params) - y_data)**2))
        curve_results['Exponential'] = {
            'rmse': exp_rmse,
            'params': exp_params,
            'func': simple_exp,
            'y': exp_y
        }
        
        # Calculate numerical derivative
        try:
            derivatives['Exponential'] = calculate_derivative(x_smooth, exp_params, simple_exp)
        except Exception as e:
            print(f"Warning: Could not calculate exponential derivative: {e}")
            derivatives['Exponential'] = np.zeros_like(x_smooth)
    except Exception as e:
        print(f"Warning: Could not fit exponential function: {e}")
    
    # If no curves could be fitted, return None
    if not curve_results:
        print("No curve models could be fitted successfully")
        return None
    
    return {
        'curves': curve_results,
        'x_smooth': x_smooth,
        'derivatives': derivatives
    }

def find_optimal_range(x_smooth, derivatives, threshold=0.5):
    """Find the range where rate of change is most significant"""
    optimal_ranges = {}
    
    for name, deriv in derivatives.items():
        # Normalize derivatives to [0,1] for easier threshold comparison
        if np.any(deriv):  # Check if derivatives aren't all zeros
            max_deriv = np.max(deriv)
            if max_deriv > 0:
                norm_deriv = deriv / max_deriv
                
                # Find points where derivative exceeds threshold
                significant_indices = np.where(norm_deriv > threshold)[0]
                if len(significant_indices) > 0:
                    start_idx = significant_indices[0]
                    end_idx = significant_indices[-1]
                    optimal_ranges[name] = (x_smooth[start_idx], x_smooth[end_idx])
                else:
                    optimal_ranges[name] = None
            else:
                optimal_ranges[name] = None
        else:
            optimal_ranges[name] = None
    
    return optimal_ranges

def plot_precision_curves(precision_results, output_dir='data'):
    """Plot precision curves for different k values"""
    # Extract dimensions and precision values
    dimensions = sorted(precision_results.keys())
    precision_at_5 = [precision_results[dim][5]['mean'] for dim in dimensions]
    precision_at_10 = [precision_results[dim][10]['mean'] for dim in dimensions]
    precision_at_15 = [precision_results[dim][15]['mean'] for dim in dimensions]
    
    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot settings
    plot_settings = {
        5: {'ax': ax1, 'color': '#1f77b4', 'title': 'Precision@5', 'values': precision_at_5},
        10: {'ax': ax2, 'color': '#2ca02c', 'title': 'Precision@10', 'values': precision_at_10},
        15: {'ax': ax3, 'color': '#d62728', 'title': 'Precision@15', 'values': precision_at_15}
    }
    
    # Create plots
    for k, settings in plot_settings.items():
        ax = settings['ax']
        values = settings['values']
        
        # Create the line plot
        ax.plot(dimensions, values,
                color=settings['color'],
                marker='o',
                linewidth=2,
                markersize=8)
        
        # Add scatter points
        ax.scatter(dimensions, values,
                  color=settings['color'],
                  s=100,
                  alpha=0.6)
        
        # Customize the plot
        ax.set_title(settings['title'], fontsize=14, pad=15)
        ax.set_xlabel('Reduced Dimension', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits slightly above and below the data range
        y_min = min(values) * 0.95
        y_max = min(1.0, max(values) * 1.05)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels on the points
        for x, y in zip(dimensions, values):
            ax.annotate(f'{y:.3f}',
                       (x, y),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center',
                       fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'precision_plots_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Precision plots saved as '{output_path}'")
    
    # Return data for further analysis
    return {
        'dimensions': dimensions,
        'precision_at_5': precision_at_5,
        'precision_at_10': precision_at_10,
        'precision_at_15': precision_at_15
    }

def plot_curve_fits(dimensions, precision_values, curve_results, x_smooth, k=10, output_dir='data'):
    """Plot curve fitting results"""
    plt.figure(figsize=(12, 8))
    
    # Plot original data points
    plt.scatter(dimensions, precision_values, color='blue', s=100, label='Data')
    
    # Plot fitted curves
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    curves = curve_results['curves']
    
    for i, (name, curve_data) in enumerate(curves.items()):
        plt.plot(x_smooth, curve_data['y'], color=colors[i % len(colors)], 
                 label=f"{name} (RMSE: {curve_data['rmse']:.4f})")
    
    # Customize plot
    plt.title(f'Precision@{k} - All Curve Fits', fontsize=16)
    plt.xlabel('Reduced Dimension', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'precision_at_{k}_curve_fits.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Curve fitting plot saved as '{output_path}'")
    
    return output_path

def plot_derivatives(dimensions, x_smooth, derivatives, k=10, output_dir='data'):
    """Plot derivatives of fitted curves to identify regions of rapid change"""
    plt.figure(figsize=(12, 8))
    
    # Plot derivatives
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for i, (name, deriv) in enumerate(derivatives.items()):
        plt.plot(x_smooth, deriv, color=colors[i % len(colors)], label=f"{name} derivative")
    
    # Customize plot
    plt.title(f'Rate of Change Analysis - Precision@{k}', fontsize=16)
    plt.xlabel('Reduced Dimension', fontsize=14)
    plt.ylabel('Rate of Change (Derivative)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'precision_at_{k}_derivatives.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Derivatives plot saved as '{output_path}'")
    
    return output_path

def plot_optimal_range(dimensions, precision_values, curve_results, optimal_ranges, k=10, output_dir='data'):
    """Plot optimal dimension range based on best fitting curve"""
    # Find best fitting model
    curves = curve_results['curves']
    x_smooth = curve_results['x_smooth']
    
    best_model = min(curves.items(), key=lambda x: x[1]['rmse'])
    best_name = best_model[0]
    best_curve = best_model[1]
    
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    plt.scatter(dimensions, precision_values, color='blue', s=100, label='Data')
    
    # Plot best fit curve
    plt.plot(x_smooth, best_curve['y'], color='red', linewidth=3, 
            label=f"{best_name} fit (RMSE: {best_curve['rmse']:.4f})")
    
    # Highlight optimal range if available
    if best_name in optimal_ranges and optimal_ranges[best_name] is not None:
        start, end = optimal_ranges[best_name]
        mask = (x_smooth >= start) & (x_smooth <= end)
        plt.fill_between(x_smooth[mask], 0, best_curve['y'][mask], 
                        color='green', alpha=0.3, 
                        label=f"Optimal range: {start:.1f} to {end:.1f}")
        
        # Add annotations for inflection points
        plt.axvline(x=start, color='green', linestyle='--')
        plt.axvline(x=end, color='green', linestyle='--')
        plt.text(start, min(precision_values) * 0.98, f"{start:.1f}", 
                rotation=90, verticalalignment='bottom')
        plt.text(end, min(precision_values) * 0.98, f"{end:.1f}", 
                rotation=90, verticalalignment='bottom')
    
    # Customize plot
    plt.title(f'Precision@{k} with Optimal Dimension Range', fontsize=16)
    plt.xlabel('Reduced Dimension', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'optimal_range_analysis_k{k}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Optimal range analysis plot saved as '{output_path}'")
    
    return output_path

def save_precision_results(dimensions, precision_results, output_file='data/precision_detailed.csv'):
    """Save detailed precision results to CSV"""
    # Create a DataFrame with the required format
    data = {
        'reduced_dimension': dimensions,
        'precision_at_5': [precision_results[dim][5]['mean'] for dim in dimensions],
        'precision_at_10': [precision_results[dim][10]['mean'] for dim in dimensions],
        'precision_at_15': [precision_results[dim][15]['mean'] for dim in dimensions]
    }
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Detailed precision results saved to '{output_file}'")
    
    return df

def save_variance_results(reduced_embeddings, output_file='data/variance.csv'):
    """Save variance data to CSV"""
    dimensions = sorted(reduced_embeddings.keys())
    data = {
        'dimension': dimensions,
        'image_variance': [reduced_embeddings[dim]['image_variance'] for dim in dimensions],
        'caption_variance': [reduced_embeddings[dim]['caption_variance'] for dim in dimensions]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Variance results saved to '{output_file}'")
    
    return df

def plot_variance(reduced_embeddings, output_dir='data'):
    """Plot explained variance vs dimension"""
    dimensions = sorted(reduced_embeddings.keys())
    image_variance = [reduced_embeddings[dim]['image_variance'] for dim in dimensions]
    caption_variance = [reduced_embeddings[dim]['caption_variance'] for dim in dimensions]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(dimensions, image_variance, 'b-', linewidth=2, marker='o', label='Image Embeddings')
    plt.plot(dimensions, caption_variance, 'r-', linewidth=2, marker='o', label='Caption Embeddings')
    
    plt.title('Explained Variance vs Dimension', fontsize=16)
    plt.xlabel('Dimensions', fontsize=14)
    plt.ylabel('Explained Variance Ratio (Cumulative)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add reference lines at common thresholds
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    
    plt.text(max(dimensions)*0.02, 0.81, '80% variance', fontsize=10)
    plt.text(max(dimensions)*0.02, 0.91, '90% variance', fontsize=10)
    plt.text(max(dimensions)*0.02, 0.96, '95% variance', fontsize=10)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'variance_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Variance plot saved as '{output_path}'")
    
    return output_path

def main():
    """Main function to run the complete analysis pipeline"""
    try:
        # Define dimensions to analyze (every 10 dimensions)
        max_dim = 320
        dimensions = list(range(10, max_dim + 1, 10))
        # Also include the data points you already have
        existing_dims = [1, 2, 4, 8, 16, 20, 32, 40, 50, 64, 80, 100, 128, 140, 
                        160, 180, 200, 220, 240, 256, 280, 300, 320]
        
        # Combine and remove duplicates
        all_dims = sorted(list(set(dimensions + existing_dims)))
        print(f"Analyzing {len(all_dims)} dimensions: {all_dims}")
        
        # Process original data (update path as needed)
        df, original_image_embeddings, original_caption_embeddings = process_original_data('data/embedding.csv')
        
        # Generate reduced embeddings
        reduced_embeddings = generate_reduced_embeddings(
            original_image_embeddings, original_caption_embeddings, all_dims
        )
        
        # Save variance results
        save_variance_results(reduced_embeddings)
        
        # Plot variance
        plot_variance(reduced_embeddings)
        
        # Calculate precision metrics
        precision_results = calculate_precision_metrics(
            original_image_embeddings, original_caption_embeddings, reduced_embeddings
        )
        
        # Save detailed precision results
        save_precision_results(all_dims, precision_results)
        
        # Plot precision curves
        plot_data = plot_precision_curves(precision_results)
        
        # Process curve fitting for each precision@k
        k_values = [5, 10, 15]
        
        for k in k_values:
            try:
                dimensions = plot_data['dimensions']
                precision_values = plot_data[f'precision_at_{k}']
                
                # Fit curves
                curve_results = fit_curves(dimensions, precision_values)
                
                if curve_results is not None:
                    # Plot curve fits
                    plot_curve_fits(dimensions, precision_values, curve_results, curve_results['x_smooth'], k)
                    
                    # Calculate optimal ranges
                    optimal_ranges = find_optimal_range(curve_results['x_smooth'], curve_results['derivatives'])
                    
                    # Plot derivatives
                    plot_derivatives(dimensions, curve_results['x_smooth'], curve_results['derivatives'], k)
                    
                    # Plot optimal range
                    plot_optimal_range(dimensions, precision_values, curve_results, optimal_ranges, k)
                    
                    # Print results
                    print(f"\nResults for Precision@{k}:")
                    
                    # Find best fitting model
                    curves = curve_results['curves'] 
                    best_model = min(curves.items(), key=lambda x: x[1]['rmse'])
                    print(f"Best fitting model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
                    
                    # Print equation for each model
                    print("\nFitted equations:")
                    for name, curve_data in curves.items():
                        if name == 'Exponential':
                            a, b, c = curve_data['params']
                            print(f"  Exponential: f(x) = {a:.4f} * (1 - exp(-{b:.6f} * x)) + {c:.4f}")
                        elif name == 'Log':
                            a, b = curve_data['params']
                            print(f"  Logarithmic: f(x) = {a:.4f} * log(x + 1) + {b:.4f}")
                        elif name == 'Logistic':
                            L, k, x0 = curve_data['params']
                            print(f"  Logistic: f(x) = {L:.4f} / (1 + exp(-{k:.6f} * (x - {x0:.4f})))")
                    
                    # Print optimal ranges
                    print("\nOptimal Dimension Ranges (where rate of change is most significant):")
                    for name, range_vals in optimal_ranges.items():
                        if range_vals:
                            print(f"  {name}: {range_vals[0]:.1f} to {range_vals[1]:.1f}")
                        else:
                            print(f"  {name}: No significant range found")
                else:
                    print(f"Curve fitting failed for Precision@{k}")
            except Exception as e:
                print(f"Error processing Precision@{k}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\nAnalysis complete!")
    except Exception as e:
        print(f"Error in main analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()