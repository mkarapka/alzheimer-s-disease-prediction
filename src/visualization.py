import matplotlib.pyplot as plt
import seaborn as sns


def plot_optimization_results(study, title="Optimization Results"):
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]

    plt.figure(figsize=(10, 6))
    plt.plot(trials_df["number"], trials_df["value"], marker="o", linestyle="-")
    plt.title(f"{title} - Objective Value per Trial")
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value (e.g., Accuracy)")
    plt.grid()
    plt.show()

    # Filter only numeric parameter columns for pairplot
    param_cols = [col for col in trials_df.columns if col.startswith("params_")]
    
    # Filter only numeric columns
    numeric_cols = [col for col in param_cols if trials_df[col].dtype in ['int64', 'float64']]
    
    # If there are more than 1 numeric parameters, create pairplot
    if len(numeric_cols) > 1:
        sns.pairplot(trials_df, vars=numeric_cols, hue="value", palette="coolwarm", diag_kind="kde")
        plt.suptitle(f"{title} - Parameter Interactions", y=1.02)
        plt.show()

    # If there are numeric columns, we can plot a correlation heatmap
    if len(numeric_cols) > 1:
        corr_matrix = trials_df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"{title} - Parameter Correlations")
        plt.show()



def plot_best_parameters(study, title="Best Parameters Distribution"):
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best Parameters: {best_params}")
    print(f"Best Value: {best_value}")

    param_cols = [col for col in trials_df.columns if col.startswith("params_")]
    for param in param_cols:
        plt.figure(figsize=(8, 6))
        
        # Update the boxplot to prevent the FutureWarning
        sns.boxplot(x=trials_df[param], y=trials_df["value"], palette="pastel", hue=trials_df[param], legend=False)
        
        plt.title(f"{title} - {param}")
        plt.ylabel("Objective Value")
        plt.xlabel(param)
        plt.grid()
        plt.show()
