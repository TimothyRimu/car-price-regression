"""Car price regression (205 cars, 24 features). Small data => repeated CV, baselines, brand-leakage check."""
import pandas as pd, numpy as np, json, sys
from sklearn.model_selection import RepeatedKFold, GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
D = sys.argv[1] if len(sys.argv) > 1 else 'data/CarPrice.csv'
c = pd.read_csv(D)
fix = {'maxda':'mazda','porcshce':'porsche','toyouta':'toyota','vokswagen':'volkswagen','vw':'volkswagen'}
c['brand'] = c.CarName.str.split().str[0].str.lower().replace(fix)
c['symboling'] = c.symboling.astype(str)
y = c.price
X = c.drop(columns=['price','car_ID','CarName'])
cat = X.select_dtypes('object').columns.tolist(); num = [k for k in X.columns if k not in cat]
def pre(drop_brand=False):
    cc = [k for k in cat if not (drop_brand and k=='brand')]
    return ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),cc),('n',StandardScaler(),num)])
def log(m): return TransformedTargetRegressor(m, func=np.log, inverse_func=np.exp)
M = lambda db: {
 'baseline: predict mean price': DummyRegressor(),
 'linear regression': make_pipeline(pre(db), LinearRegression()),
 'ridge (log price)': log(make_pipeline(pre(db), RidgeCV(alphas=np.logspace(-2,3,20)))),
 'lasso (log price)': log(make_pipeline(pre(db), LassoCV(cv=5, max_iter=20000, random_state=0))),
 'random forest': make_pipeline(pre(db), RandomForestRegressor(300, random_state=0)),
}
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
sc = {'r2':'r2','mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error'}
out = {}
for name, db in (('with brand', False), ('without brand', True)):
    out[name] = {}
    for k, m in M(db).items():
        r = cross_validate(m, X, y, cv=cv, scoring=sc, n_jobs=1)
        out[name][k] = {'R2': float(r['test_r2'].mean()), 'R2_sd': float(r['test_r2'].std()), 'MAE': float(-r['test_mae'].mean()), 'RMSE': float(-r['test_rmse'].mean())}
# unseen-brand test: does the model transfer to makes it has never seen?
g = GroupKFold(5); gm = {}
for k, m in M(False).items():
    p = cross_val_predict(m, X, y, cv=g, groups=c.brand)
    gm[k] = {'R2': float(1-((y-p)**2).sum()/((y-y.mean())**2).sum()), 'MAE': float(abs(y-p).mean())}
out['unseen brands (GroupKFold by brand)'] = gm
# what drives price (linear model, standardised numerics only)
rf = make_pipeline(pre(True), RandomForestRegressor(300, random_state=0)).fit(X, y)
names = rf[0].get_feature_names_out(); imp = pd.Series(rf[1].feature_importances_, names).sort_values(ascending=False)
out['rf_top_features'] = {k.replace('n__','').replace('c__',''): round(float(v),3) for k, v in imp.head(6).items()}
out['corr_with_price'] = c[num+['price']].corr().price.drop('price').abs().sort_values(ascending=False).head(5).round(3).to_dict()
json.dump(out, open('results.json','w'), indent=1)
for k, v in out.items():
    print(k)
    if isinstance(v, dict) and all(isinstance(i, dict) for i in v.values()):
        for m, s in v.items(): print(f'   {m:30s}', {a: round(b,3 if a=='R2' else 0) for a, b in s.items() if a!='R2_sd'})
    else: print('  ', v)
print(c.brand.value_counts().head(5).to_dict(), c.brand.nunique())
