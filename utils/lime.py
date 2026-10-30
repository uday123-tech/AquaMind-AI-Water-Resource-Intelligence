import os
import pandas as pd
import numpy as np
import joblib
import torch
import plotly.graph_objects as go


def build_lime_background(features, selected_features, num_cols, encoding, preprocess_raw_features_fn, background_paths=None):
    """
    Build background data for LIME explanations for HVAC predictive maintenance.
    """
    if background_paths is None:
        background_paths = [
            os.path.join("Models", "lime_background.csv"),
            os.path.join("Models", "maintenance_background.csv"),
        ]

    for background_path in background_paths:
        if os.path.exists(background_path):
            background = pd.read_csv(background_path)
            missing_cols = [col for col in selected_features if col not in background.columns]
            if not missing_cols:
                background = background.drop(columns=["Target", "failure", "anomaly"], errors="ignore")
                background = background[selected_features]
                try:
                    background = preprocess_raw_features_fn(background).dropna()
                except ValueError:
                    # Background may already be encoded (numeric categorical columns).
                    background = background.apply(pd.to_numeric, errors="coerce").dropna()
                if not background.empty:
                    return background

    # Generate synthetic background data from HVAC/maintenance dataset
    row = features.iloc[0].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    scale = np.maximum(np.abs(row) * 0.08, 0.03)
    synthetic_rows = rng.normal(loc=row, scale=scale, size=(250, len(selected_features)))
    background = pd.DataFrame(synthetic_rows, columns=selected_features)

    cat_cols = [col for col in selected_features if col not in num_cols]
    for col in cat_cols:
        if col in encoding:
            max_value = len(encoding[col].classes_) - 1
            background[col] = background[col].round().clip(0, max_value)

    return pd.concat([features, background], ignore_index=True)


def generate_lime_html(
    features,
    model,
    selected_features,
    num_cols,
    encoding,
    class_names_fn,
    preprocess_raw_features_fn,
    background_paths=None,
):
    """
    Generate LIME HTML explanation for HVAC predictive maintenance model.
    """
    if not hasattr(model, "predict_proba"):
        return None, "LIME needs a model with predict_proba."

    try:
        from lime import lime_tabular
    except ImportError:
        return None, "Install lime to display the explanation: pip install lime"

    try:
        background = build_lime_background(
            features,
            selected_features,
            num_cols,
            encoding,
            preprocess_raw_features_fn,
            background_paths=background_paths,
        )
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=background.to_numpy(dtype=float),
            feature_names=selected_features,
            class_names=class_names_fn(),
            mode="classification",
        )
        explanation = explainer.explain_instance(
            data_row=features.iloc[0].to_numpy(dtype=float),
            predict_fn=lambda rows: model.predict_proba(
                pd.DataFrame(rows, columns=selected_features)
            ),
            num_features=len(selected_features),
        )
        return explanation.as_html(), None
    except Exception as exc:
        return None, f"LIME explanation could not be generated: {exc}"


def get_hvac_feature_names():
    """Get HVAC feature names from saved config."""
    try:
        return joblib.load("Models/hvac_feature_columns.pkl")
    except:
        return ['t_supply', 't_return', 'sp_return', 't_saturation', 't_outdoor', 
                'rh_supply', 'rh_return', 'rh_outdoor', 'energy', 'power']


def get_maintenance_feature_names():
    """Get maintenance feature names from saved config."""
    try:
        return joblib.load("Models/maintenance_feature_columns.pkl")
    except:
        return ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 
                'metric6', 'metric7', 'metric8', 'metric9']


class SimplifiedModelWrapper:
    """
    Wrapper for BiLSTM-BiGRU model that keeps VAE embedding fixed for LIME.
    This allows LIME to explain only the maintenance metrics while
    keeping the HVAC embedding constant.
    """
    def __init__(self, original_model, vae_model, hvac_scaler, hvac_features, selected_date_hvac, device):
        self.original_model = original_model
        self.vae_model = vae_model
        self.hvac_scaler = hvac_scaler
        self.hvac_features = hvac_features
        self.selected_date_hvac = selected_date_hvac
        self.device = device
        self._cached_latent = None
    
    def _get_latent_embedding(self):
        """Get and cache the VAE embedding for the selected date."""
        if self._cached_latent is None:
            with torch.no_grad():
                hvac = self.selected_date_hvac[self.hvac_features].mean().values.reshape(1, -1)
                hvac_scaled = self.hvac_scaler.transform(hvac)
                latent = self.vae_model.encode(torch.FloatTensor(hvac_scaled))[0].numpy()
                self._cached_latent = latent
        return self._cached_latent
    
    def predict_proba(self, X):
        """
        Predict probabilities for perturbed maintenance features.
        X is numpy array of maintenance metrics (scaled).
        """
        with torch.no_grad():
            # Get the fixed VAE embedding
            latent = self._get_latent_embedding()
            
            # Combine with perturbed maintenance features
            combined = np.hstack([X, np.tile(latent, (len(X), 1))])
            
            prob = self.original_model(
                torch.tensor(combined, dtype=torch.float32).unsqueeze(1).to(self.device)
            ).cpu().numpy()
        return np.column_stack([1 - prob, prob])


def generate_hvac_lime_explanation(metric_values, bilstm_bigru_model, vae_model, hvac_scaler, 
                                     maint_scaler, hvac_features, selected_date_hvac, device):
    """
    Generate LIME explanation for HVAC predictive maintenance.
    
    Args:
        metric_values: List of 9 maintenance metric values
        bilstm_bigru_model: Trained BiLSTM-BiGRU model
        vae_model: Trained VAE model
        hvac_scaler: Fitted HVAC scaler
        maint_scaler: Fitted maintenance scaler
        hvac_features: List of HVAC feature names
        selected_date_hvac: DataFrame with HVAC data for selected date
        device: torch device (cpu or cuda)
    
    Returns:
        tuple: (lime_html, error_message)
    """
    try:
        from lime import lime_tabular
    except ImportError:
        return None, "Install lime to display the explanation: pip install lime"
    
    try:
        # Create model wrapper
        model_wrapper = SimplifiedModelWrapper(
            bilstm_bigru_model, vae_model, hvac_scaler, 
            hvac_features, selected_date_hvac, device
        )
        
        # Get feature names
        feature_names = get_maintenance_feature_names()
        
        # Prepare features for LIME (use scaled maintenance features)
        features_df = pd.DataFrame([metric_values], columns=feature_names)
        features_scaled = maint_scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)
        
        # Build background data
        background = build_lime_background(
            features=features_scaled_df,
            selected_features=feature_names,
            num_cols=feature_names,
            encoding={},
            preprocess_raw_features_fn=lambda x: maint_scaler.transform(x),
        )
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=background.to_numpy(dtype=float),
            feature_names=feature_names,
            class_names=["NORMAL", "FAILURE"],
            mode="classification",
        )
        
        # Generate explanation
        explanation = explainer.explain_instance(
            data_row=features_scaled_df.iloc[0].to_numpy(dtype=float),
            predict_fn=model_wrapper.predict_proba,
            num_features=len(feature_names),
        )
        
        return explanation.as_html(), None
    except Exception as exc:
        return None, f"LIME explanation could not be generated: {exc}"


def create_lime_plot(
    lime_features,
    lime_values,
    predicted_label
):
    bar_colors = [
        "green" if value >= 0 else "red"
        for value in lime_values
    ]

    figure = go.Figure(
        go.Bar(
            x=lime_values,
            y=lime_features,
            orientation="h",
            marker=dict(
                color=bar_colors
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Contribution: %{x:.6f}"
                "<extra></extra>"
            )
        )
    )

    figure.update_layout(
        title={
            "text": (
                "LIME Feature Contribution - "
                f"{predicted_label}"
            ),
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="Feature Contribution",
        yaxis_title="Feature",
        height=650,
        margin=dict(
            l=260,
            r=50,
            t=90,
            b=80
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Arial",
            size=14,
            color="#294164"
        ),
        showlegend=False
    )

    figure.update_xaxes(
        showgrid=True,
        gridcolor="#dce6f2",
        zeroline=True,
        zerolinecolor="#333333"
    )

    figure.update_yaxes(
        autorange="reversed"
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "responsive": True,
            "displaylogo": False
        }
    )


def generate_plotly_lime(
    raw_features,
    predicted_class,
    feature_names,
    class_names_fn,
    predict_fn,
    target_encoder
):
    try:
        from lime.lime_tabular import LimeTabularExplainer

        feature_row = raw_features[
            feature_names
        ].iloc[0].to_numpy(dtype=float)

        rng = np.random.default_rng(42)

        perturbation_scale = np.maximum(
            np.abs(feature_row) * 0.08,
            0.05
        )

        background_array = rng.normal(
            loc=feature_row,
            scale=perturbation_scale,
            size=(300, len(feature_names))
        )

        background_array = np.vstack([
            feature_row,
            background_array
        ])

        explainer = LimeTabularExplainer(
            training_data=background_array,
            feature_names=feature_names,
            class_names=class_names_fn(),
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )

        explanation = explainer.explain_instance(
            data_row=feature_row,
            predict_fn=predict_fn,
            labels=[predicted_class],
            num_features=len(feature_names),
            num_samples=1000
        )

        lime_data = explanation.as_list(
            label=predicted_class
        )

        lime_features = [
            feature
            for feature, contribution in lime_data
        ]

        lime_values = [
            contribution
            for feature, contribution in lime_data
        ]

        predicted_label = target_encoder.inverse_transform(
            [predicted_class]
        )[0]

        lime_plot = create_lime_plot(
            lime_features=lime_features,
            lime_values=lime_values,
            predicted_label=predicted_label
        )

        return lime_plot, None

    except Exception as error:
        return None, str(error)
