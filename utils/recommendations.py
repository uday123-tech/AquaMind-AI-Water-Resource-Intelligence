import json
import socket
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"
GENERATION_TIMEOUT_SECONDS = 180

def get_badge_class(suitability_str):
    s = str(suitability_str).lower()
    if "high" in s or "excellent" in s:
        return "success"
    elif "recom" in s or "good" in s or "primary" in s:
        return "primary"
    elif "mod" in s or "warn" in s or "fair" in s:
        return "warning"
    elif "toleran" in s or "salt" in s:
        return "secondary"
    return "info"

def get_fallback_content(prediction_label, is_water_quality=True, features=None):
    lbl = str(prediction_label).upper()
    features = features or {}

    if not is_water_quality:
        if "HIGH" in lbl:
            return {
                "recommendations": [
                    "Promote sustainable water consumption practices to preserve the current high availability levels.",
                    "Maintain and clean rainwater harvesting structures to capture seasonal rainfall peaks.",
                    "Optimize reservoir release schedules to support agricultural and domestic needs without depleting stocks.",
                    "Monitor groundwater recharge rates to ensure long-term aquifer health."
                ],
                "suggestions": [
                    "Educate the public on the importance of water conservation even when availability is high.",
                    "Conduct regular audits of industrial water usage to ensure compliance with efficiency standards.",
                    "Upgrade municipal distribution channels to prevent infrastructure leaks.",
                    "Perform seasonal surveys of water body volumes to track resource replenishment."
                ],
                "crops": []
            }
        elif "LOW" in lbl:
            return {
                "recommendations": [
                    "Implement immediate water conservation measures and restrict non-essential water usage.",
                    "Deploy micro-irrigation systems (such as drip or sprinkler) in agricultural fields to reduce waste.",
                    "Perform routine leak detection and repair audits across the distribution network.",
                    "Establish emergency water sharing agreements with neighboring districts to meet domestic demand."
                ],
                "suggestions": [
                    "Distribute low-flow aerators and water-saving devices to residential areas.",
                    "Utilize recycled greywater for landscaping, public parks, and non-potable industrial uses.",
                    "Implement temporary price adjustments to discourage excessive water consumption.",
                    "Schedule emergency dredging of dry reservoir beds to maximize capacity for future rains."
                ],
                "crops": []
            }
        else:  # MEDIUM
            return {
                "recommendations": [
                    "Adopt micro-irrigation systems (drip or sprinkler) to maximize water use efficiency.",
                    "Monitor soil salinity and moisture regularly to prevent depletion in the root zone.",
                    "Schedule irrigation during cooler hours to minimize evaporative losses.",
                    "Evaluate potential upstream runoff sources to optimize distribution."
                ],
                "suggestions": [
                    "Install basic filtration or settling tanks to reduce suspended turbidity before irrigation.",
                    "Incorporate organic matter and biochar into the soil to retain soil moisture.",
                    "Perform routine leak checks across municipal channels.",
                    "Conduct quarterly water storage checks to track reserve levels."
                ],
                "crops": []
            }

    # Water Quality Prediction Fallback Content
    tds = features.get("TDS_mgL", 150)
    ph = features.get("pH", 7.0)

    if "EXCELLENT" in lbl:
        return {
            "recommendations": [
                "Ensure periodic testing remains scheduled to monitor for any seasonal variation in water parameters.",
                "Maintain current watershed protection policies and prevent any new source of runoff nearby.",
                "Document the current physical and chemical baseline to track future changes in quality.",
                "Promote community awareness about preserving local water resources and keeping them pollutant-free."
            ],
            "suggestions": [
                "Verify that water treatment systems, if any, are operating at baseline energy levels.",
                "Keep records of Dissolved Oxygen and pH levels during high-flow seasons.",
                "Promote rainwater harvesting and sustainable irrigation methods in the surrounding area.",
                "Assess upstream agricultural practices regularly to proactively manage fertilizer runoff."
            ],
            "crops": [
                {
                    "name": "Paddy Rice",
                    "category": "Cereal Grain",
                    "suitability": "Highly Recommended",
                    "badge_class": "success",
                    "reason": f"With pure water quality ({lbl}), low TDS ({tds} mg/L), and optimal oxygenation, high-yield flood and furrow irrigation can be sustained without root toxicity."
                },
                {
                    "name": "Tomato & Bell Pepper",
                    "category": "Horticulture / Vegetables",
                    "suitability": "Highly Recommended",
                    "badge_class": "success",
                    "reason": f"Sensitive to salinity and chemical contaminants; thrives with the balanced pH ({ph}) and low TDS water profile."
                },
                {
                    "name": "Sugarcane",
                    "category": "Cash Crop",
                    "suitability": "Recommended",
                    "badge_class": "primary",
                    "reason": "Water-intensive crop that produces superior sucrose content when irrigated with pollutant-free water."
                },
                {
                    "name": "Strawberries & Exotic Berries",
                    "category": "High-Value Fruits",
                    "suitability": "Highly Recommended",
                    "badge_class": "success",
                    "reason": "Delicate shallow root systems demand high dissolved oxygen and low mineral residues for optimal growth."
                }
            ]
        }
    elif "GOOD" in lbl:
        return {
            "recommendations": [
                "Maintain current watershed protection policies and monitor seasonal parameter shifts.",
                "Continue standard soil testing to ensure water-soil nutrient balance.",
                "Optimize irrigation cycles to prevent unnecessary water runoff.",
                "Promote sustainable agricultural practices to maintain current water safety."
            ],
            "suggestions": [
                "Keep records of Dissolved Oxygen and pH levels during high-flow seasons.",
                "Inspect irrigation channels periodically for sediment buildup.",
                "Promote organic mulching to conserve moisture in agricultural fields.",
                "Schedule routine water audits before peak planting seasons."
            ],
            "crops": [
                {
                    "name": "Wheat",
                    "category": "Cereal Grain",
                    "suitability": "Highly Recommended",
                    "badge_class": "success",
                    "reason": f"Excellent yield potential under Good water quality, tolerating moderate mineral levels with balanced pH ({ph})."
                },
                {
                    "name": "Maize (Corn)",
                    "category": "Cereal Grain",
                    "suitability": "Recommended",
                    "badge_class": "primary",
                    "reason": "Responds well to steady moisture with safe dissolved oxygen and low biological oxygen demand."
                },
                {
                    "name": "Pulses (Chickpea & Lentils)",
                    "category": "Legumes",
                    "suitability": "Recommended",
                    "badge_class": "primary",
                    "reason": "Effective root nodulation and nitrogen fixation thrive under standard neutral water conditions."
                },
                {
                    "name": "Mustard & Groundnut",
                    "category": "Oilseeds",
                    "suitability": "Suitable",
                    "badge_class": "info",
                    "reason": "Performs reliably under good irrigation water with low biological oxygen demand and controlled turbidity."
                }
            ]
        }
    elif "MODERATE" in lbl or "MEDIUM" in lbl:
        return {
            "recommendations": [
                "Adopt micro-irrigation systems (drip or sprinkler) to maximize water use efficiency and prevent salt concentration.",
                "Monitor soil salinity and TDS regularly to prevent mineral buildup in the root zone.",
                "Schedule irrigation during cooler hours to minimize evaporative losses.",
                "Evaluate potential upstream runoff sources to prevent further parameter degradation."
            ],
            "suggestions": [
                "Install basic filtration or settling tanks to reduce suspended turbidity before irrigation.",
                "Incorporate organic matter and biochar into the soil to buffer moderate water chemistry.",
                "Rotate salt-sensitive crops with hardier cereal varieties.",
                "Conduct quarterly water tests to track changes in BOD and Nitrate levels."
            ],
            "crops": [
                {
                    "name": "Cotton",
                    "category": "Fiber / Cash Crop",
                    "suitability": "Recommended",
                    "badge_class": "primary",
                    "reason": f"Deep root architecture and natural salt tolerance allow it to handle moderate TDS ({tds} mg/L) and moderate water quality without yield degradation."
                },
                {
                    "name": "Barley",
                    "category": "Cereal Grain",
                    "suitability": "Highly Recommended",
                    "badge_class": "success",
                    "reason": f"Recognized as one of the most salt-tolerant cereal grains; performs reliably in moderate water conditions and elevated mineral content."
                },
                {
                    "name": "Pearl Millet (Bajra) & Sorghum",
                    "category": "Millets & Coarse Grains",
                    "suitability": "Highly Recommended",
                    "badge_class": "success",
                    "reason": "Exceptionally hardy with high nutrient-use efficiency, thriving in moderate water quality and higher turbidity."
                },
                {
                    "name": "Sunflower & Safflower",
                    "category": "Oilseeds",
                    "suitability": "Suitable",
                    "badge_class": "warning",
                    "reason": "Strong taproot system efficiently utilizes moderate water quality with consistent seed oil yield."
                }
            ]
        }
    else:  # POOR / Contaminated
        return {
            "recommendations": [
                "Investigate primary contamination sources, focusing on BOD, nitrate, and turbidity levels.",
                "Implement filtration or chemical treatment (such as aeration or disinfection) before irrigation.",
                "Avoid irrigating fresh vegetable or food crops that are consumed raw with this water.",
                "Report significant deviations in water safety indexes to local environmental authorities."
            ],
            "suggestions": [
                "Consider agricultural buffer zones to intercept fertilizer and sediment runoff before they reach water bodies.",
                "Optimize aeration systems if dissolved oxygen drops below acceptable biological thresholds.",
                "Inspect nearby sewage and wastewater discharge channels for potential leaks or structural failures.",
                "Install turbidity barriers in areas where construction or soil erosion affects water clarity."
            ],
            "crops": [
                {
                    "name": "Sugar Beet",
                    "category": "Industrial Root Crop",
                    "suitability": "Salt-Tolerant",
                    "badge_class": "warning",
                    "reason": f"Exceptional physiological tolerance to elevated salinity, high TDS ({tds} mg/L), and poor water quality."
                },
                {
                    "name": "Vetiver / Rhodes Grass",
                    "category": "Forage & Phytoremediation",
                    "suitability": "Recommended for Remediation",
                    "badge_class": "primary",
                    "reason": "Deep-root phytoremediation grass that absorbs excess nitrates and organic pollutants while stabilizing the soil."
                },
                {
                    "name": "Industrial Hemp & Flax",
                    "category": "Commercial Fiber (Non-Food)",
                    "suitability": "Tolerant",
                    "badge_class": "secondary",
                    "reason": "Resilient non-food fiber crop capable of growing with stressed or wastewater irrigation."
                },
                {
                    "name": "Casuarina & Eucalyptus",
                    "category": "Agro-Forestry / Timber",
                    "suitability": "Highly Tolerant",
                    "badge_class": "secondary",
                    "reason": "Hardy timber species that withstand high TDS, nitrates, and degraded water conditions."
                }
            ]
        }

def build_prompt(features, prediction_label, confidence_score, is_water_quality=True):
    feature_lines = "\n".join(
        f"- {name}: {value}" for name, value in features.items()
    )

    if not is_water_quality:
        # Prompt for water availability forecasting only (no crop recommendations)
        return f"""
You are an expert environmental scientist and water systems analyst. Generate key water conservation recommendations and proactive mitigation suggestions based on the water availability forecast result: {prediction_label}.

Return a JSON object with exactly two keys: "recommendations" and "suggestions". Each value must be a list of exactly 4 clear, practical, and distinct sentences.

Water Forecast Details:
- Forecasted Availability Classification: {prediction_label}
- Model Accuracy / Confidence: {confidence_score}%
- Input Profile:
{feature_lines}
""".strip()

    # Dedicated Prompt for Water Quality Prediction with Crop Recommendation strictly based on user inputs + prediction result
    return f"""
You are an expert agricultural scientist, agronomist, and irrigation water specialist.
The user has provided specific water quality test parameters, and our Machine Learning model predicted the overall Water Quality Classification result.

YOUR OBJECTIVE:
Generate 4 key water recommendations, 4 proactive suggestions, and 4 specific agricultural crop recommendations that can be successfully grown based strictly on:
1. The USER INPUT PARAMETERS submitted in the Water Quality Prediction section:
   - pH: {features.get('pH')} (measures acidity/alkalinity)
   - TDS (Total Dissolved Solids): {features.get('TDS_mgL')} mg/L (measures salinity and dissolved minerals)
   - Turbidity: {features.get('Turbidity_NTU')} NTU (measures suspended particles and clarity)
   - Dissolved Oxygen (DO): {features.get('Dissolved_Oxygen_mgL')} mg/L (measures oxygen availability)
   - BOD (Biochemical Oxygen Demand): {features.get('BOD_mgL')} mg/L (measures organic pollutant load)
   - Nitrate: {features.get('Nitrate_mgL')} mg/L (measures nitrogen concentration)
   - Rainfall: {features.get('Rainfall_mm')} mm
   - River Flow: {features.get('River_Flow_m3s')} m3/s
   - Season: {features.get('Season')}
   - Humidity: {features.get('Humidity_pct')}%
   - Calculated Pollution Index: {features.get('Pollution_Index')}
   - Calculated Water Quality Score: {features.get('Water_Quality_Score')}
2. The ML PREDICTED WATER QUALITY RESULT:
   - Classification: {prediction_label}
   - Model Confidence: {confidence_score}%

CRITICAL INSTRUCTIONS FOR CROP GENERATION:
- Recommend 4 crops specifically suited to this water quality profile ({prediction_label}) and the exact input parameters (such as the given TDS of {features.get('TDS_mgL')} mg/L and pH of {features.get('pH')}).
- In the "reason" field for each crop, EXPLICITLY explain how the crop tolerates or thrives with the user's specific water quality parameters (e.g. salinity tolerance for the given TDS, pH suitability, BOD/turbidity tolerance, or nutrient uptake).

Return a JSON object with exactly three keys: "recommendations", "suggestions", and "crops".
- "recommendations": List of 4 distinct sentences for water quality treatment, conservation, and testing.
- "suggestions": List of 4 distinct proactive suggestions.
- "crops": List of 4 crop objects, each having:
  - "name": Crop Name (e.g. "Barley", "Cotton", "Pearl Millet (Bajra)", "Sunflower", "Paddy Rice", "Wheat", "Sugar Beet")
  - "category": Crop Category (e.g. "Cereal Grain", "Fiber / Cash Crop", "Millets & Coarse Grains", "Oilseeds", "Horticulture / Vegetables", "Industrial / Remediation")
  - "suitability": Rating ("Highly Recommended", "Recommended", "Suitable", "Salt-Tolerant")
  - "reason": 1-2 sentences directly linking the crop's cultivation suitability to the user's water quality inputs (pH, TDS, Turbidity, DO, BOD) and the predicted result ({prediction_label}).

Example JSON structure:
{{
  "recommendations": [
    "Schedule regular testing to monitor key chemical parameters like Nitrate and TDS.",
    "Implement chemical aeration if dissolved oxygen levels fall below acceptable ranges.",
    "Conduct upstream inspections to locate potential agricultural fertilizer runoff.",
    "Enforce strict local regulations regarding industrial wastewater discharge."
  ],
  "suggestions": [
    "Consider installing secondary filtration units to handle elevated turbidity levels.",
    "Plant riparian vegetation along the riverbanks to filter sediment and runoff.",
    "Educate the community on sustainable water usage and pollution prevention.",
    "Store water samples from high-rainfall periods for deep laboratory validation."
  ],
  "crops": [
    {{
      "name": "Barley",
      "category": "Cereal Grain",
      "suitability": "Highly Recommended",
      "reason": "Exhibits strong natural salinity tolerance, performing reliably with a TDS level of {features.get('TDS_mgL')} mg/L and {prediction_label} water quality."
    }},
    {{
      "name": "Cotton",
      "category": "Fiber / Cash Crop",
      "suitability": "Recommended",
      "reason": "Deep root architecture allows it to thrive in moderate water quality and withstand slightly elevated mineral content."
    }},
    {{
      "name": "Pearl Millet (Bajra)",
      "category": "Millets & Coarse Grains",
      "suitability": "Highly Recommended",
      "reason": "Highly resilient to moderate turbidity and fluctuating water pH, producing dependable grain yields."
    }},
    {{
      "name": "Sunflower",
      "category": "Oilseeds",
      "suitability": "Suitable",
      "reason": "Tolerates moderate irrigation water parameters effectively while maintaining high oil seed productivity."
    }}
  ]
}}
""".strip()

def clean_item(text):
    blocked_phrases = [
        "the model suggests",
        "the model indicates",
        "the model's output",
        "the model output",
        "model output",
        "the model",
    ]
    cleaned = str(text).strip().lstrip("-*0123456789. ")
    for phrase in blocked_phrases:
        cleaned = cleaned.replace(phrase, "the result")
        cleaned = cleaned.replace(phrase.title(), "The result")
    return cleaned

def parse_llm_content(text, prediction_label, is_water_quality=True, features=None):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]

    content = json.loads(text)

    rec_key = None
    sug_key = None
    crop_key = None
    for key in content.keys():
        k_lower = key.lower()
        if k_lower == "recommendations":
            rec_key = key
        elif k_lower == "suggestions":
            sug_key = key
        elif "crop" in k_lower:
            crop_key = key

    raw_recs = content.get(rec_key) if rec_key else []
    raw_sugs = content.get(sug_key) if sug_key else []
    raw_crops = content.get(crop_key) if crop_key else []

    if isinstance(raw_recs, str):
        raw_recs = [raw_recs]
    if isinstance(raw_sugs, str):
        raw_sugs = [raw_sugs]
    if isinstance(raw_crops, str):
        raw_crops = [raw_crops]

    recommendations = [
        clean_item(item) for item in raw_recs if str(item).strip()
    ]
    suggestions = [
        clean_item(item) for item in raw_sugs if str(item).strip()
    ]

    crops = []
    if is_water_quality and isinstance(raw_crops, list):
        for item in raw_crops:
            if isinstance(item, dict):
                name = clean_item(item.get("name", item.get("crop", "Agricultural Crop")))
                category = clean_item(item.get("category", item.get("type", "Agricultural Crop")))
                suitability = clean_item(item.get("suitability", "Recommended"))
                reason = clean_item(item.get("reason", item.get("description", item.get("details", ""))))
                badge = get_badge_class(suitability)
                if name:
                    crops.append({
                        "name": name,
                        "category": category,
                        "suitability": suitability,
                        "badge_class": badge,
                        "reason": reason
                    })
            elif isinstance(item, str) and item.strip():
                clean_str = clean_item(item)
                parts = clean_str.split(":", 1) if ":" in clean_str else clean_str.split("-", 1)
                if len(parts) == 2:
                    crop_name = parts[0].strip()
                    crop_reason = parts[1].strip()
                else:
                    crop_name = clean_str
                    crop_reason = f"Suitable for cultivation under {prediction_label} water quality conditions."
                crops.append({
                    "name": crop_name,
                    "category": "Agricultural Crop",
                    "suitability": "Recommended",
                    "badge_class": "primary",
                    "reason": crop_reason
                })

    fallback_data = get_fallback_content(prediction_label, is_water_quality=is_water_quality, features=features)
    if is_water_quality and not crops:
        crops = fallback_data.get("crops", [])
    if not recommendations:
        recommendations = fallback_data.get("recommendations", [])
    if not suggestions:
        suggestions = fallback_data.get("suggestions", [])

    return {
        "recommendations": recommendations,
        "suggestions": suggestions,
        "crops": crops if is_water_quality else [],
        "suitable_crops": crops if is_water_quality else [],
        "source": "LLM",
        "model": OLLAMA_MODEL,
    }

def fallback_result(warning, prediction_label, is_water_quality=True, features=None):
    fallback_data = get_fallback_content(prediction_label, is_water_quality=is_water_quality, features=features)
    return {
        **fallback_data,
        "suitable_crops": fallback_data.get("crops", []) if is_water_quality else [],
        "crops": fallback_data.get("crops", []) if is_water_quality else [],
        "source": "Fallback",
        "model": None,
    }, warning

def generate_recommendations(features, prediction_label, confidence_score, is_water_quality=True):
    prompt = build_prompt(features, prediction_label, confidence_score, is_water_quality=is_water_quality)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=GENERATION_TIMEOUT_SECONDS,
        ) as response:
            result = json.loads(response.read().decode("utf-8"))

        recommendation_text = result.get("response", "").strip()
        if recommendation_text:
            try:
                return parse_llm_content(recommendation_text, prediction_label, is_water_quality=is_water_quality, features=features), None
            except Exception:
                return fallback_result(
                    "Ollama returned text that could not be parsed, so default guidance is shown.",
                    prediction_label,
                    is_water_quality=is_water_quality,
                    features=features
                )

        return fallback_result(
            "Ollama returned an empty response, so default guidance is shown.",
            prediction_label,
            is_water_quality=is_water_quality,
            features=features
        )
    except socket.timeout:
        return fallback_result(
            "Ollama generation timed out, so default guidance is shown.",
            prediction_label,
            is_water_quality=is_water_quality,
            features=features
        )
    except urllib.error.URLError as exc:
        return fallback_result(
            f"Could not connect to Ollama ({exc.reason}), so default guidance is shown.",
            prediction_label,
            is_water_quality=is_water_quality,
            features=features
        )
    except Exception as exc:
        return fallback_result(
            f"Recommendation generation failed ({exc}), so default guidance is shown.",
            prediction_label,
            is_water_quality=is_water_quality,
            features=features
        )
