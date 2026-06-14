from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df, features):
    cat_features = df[features].select_dtypes(include="object").columns.tolist()
    num_features = [c for c in features if c not in cat_features]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )


def build_classifier(kind="gb"):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
    if kind == "extra":
        return ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced")
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    if kind == "tree_shallow":
        return DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced")
    return GradientBoostingClassifier(random_state=42)


def build_pipeline(df, features, kind="gb"):
    return Pipeline([
        ("preprocess", build_preprocessor(df, features)),
        ("classifier", build_classifier(kind)),
    ])
