import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import xgboost as xgboost
from imblearn.pipeline import Pipeline
from scipy.special import expit
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.neighbors import NearestNeighbors


def main():
    # URL = r"D:\Users\laien\Documents\openClassRoom\P7_WU_laien\dashboard"
    URL = '.'

    def file_selector(folder_path):
        filenames = os.listdir(folder_path)
        filenames = [filename for filename in filenames if filename.endswith(".csv")]
        selected_filename = st.selectbox('Select your data', filenames)
        return os.path.join(folder_path, selected_filename)

    @st.cache
    def get_sk_id_list(df):
        # SK_IDS = df['SK_ID_CURR'].sample(1000).tolist()
        SK_IDS = df['SK_ID_CURR'].tolist()
        return SK_IDS

    @st.cache
    def get_features_description():
        tmp = pd.read_csv(URL + '/data/HomeCredit_columns_description.csv', encoding='unicode_escape')
        tmp = tmp.rename(columns={'Row': 'Features'})
        return tmp[['Features', 'Description']]

    @st.cache
    def get_model(path):
        return pd.read_pickle(path)

    @st.cache
    def get_feature_out(estimator, feature_in):
        if hasattr(estimator, 'get_feature_names'):
            if isinstance(estimator, _VectorizerMixin):
                # handling all vectorizers
                return [f'vec_{f}' \
                        for f in estimator.get_feature_names()]
            else:
                return estimator.get_feature_names(feature_in)
        elif isinstance(estimator, SelectorMixin):
            return np.array(feature_in)[estimator.get_support()]
        else:
            return feature_in

    @st.cache
    def get_ct_feature_names(ct):
        # handles all estimators, pipelines inside ColumnTransfomer
        # doesn't work when remainder =='passthrough'
        # which requires the input column names.
        output_features = []

        for name, estimator, features in ct.transformers_:
            if name != 'remainder':
                if isinstance(estimator, Pipeline):
                    current_features = features
                    for step in estimator:
                        current_features = get_feature_out(step, current_features)
                    features_out = current_features
                else:
                    features_out = get_feature_out(estimator, features)
                output_features.extend(features_out)
            elif estimator == 'passthrough':
                output_features.extend(ct._feature_names_in[features])

        return output_features

    @st.cache
    def plot_boxplot_var_by_target(X_all, y_all, X_neigh, y_neigh, X_cust,
                                   main_cols, figsize=(15, 4)):
        df_all = pd.concat([X_all[main_cols], y_all.to_frame(name='TARGET')], axis=1)
        df_neigh = pd.concat([X_neigh[main_cols], y_neigh.to_frame(name='TARGET')], axis=1)
        df_cust = X_cust[main_cols].to_frame('values').reset_index()

        fig, ax = plt.subplots(figsize=figsize)

        # random sample of customers of the train set
        df_melt_all = df_all.reset_index()
        df_melt_all.columns = ['index'] + list(df_melt_all.columns)[1:]
        df_melt_all = df_melt_all.melt(id_vars=['index', 'TARGET'],  # SK_ID_CURR
                                       value_vars=main_cols,
                                       var_name="variables",
                                       value_name="values")
        sns.boxplot(data=df_melt_all, x='variables', y='values', hue='TARGET', linewidth=1,
                    width=0.4, palette=['tab:green', 'tab:red'], showfliers=False, saturation=0.5,
                    ax=ax)

        # 20 nearest neighbors
        df_melt_neigh = df_neigh.reset_index()
        df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
        df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],  # SK_ID_CURR
                                           value_vars=main_cols,
                                           var_name="variables",
                                           value_name="values")
        sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                      palette=['darkgreen', 'darkred'], marker='o', edgecolor='k', ax=ax)

        # applicant customer
        df_melt_cust = df_cust.rename(columns={'index': "variables"})
        sns.swarmplot(data=df_melt_cust, x='variables', y='values', linewidth=1, color='y',
                      marker='o', size=10, edgecolor='k', label='applicant customer', ax=ax)

        # legend
        h, _ = ax.get_legend_handles_labels()
        ax.legend(handles=h[:5])

        plt.xticks(rotation=20)

        return fig

    @st.cache
    def shap_transform_scale(shap_values, expected_value, model_prediction):
        expected_value_transformed = expit(expected_value)
        original_explanation_distance = sum(shap_values)
        distance_to_explain = model_prediction - expected_value_transformed
        distance_coefficient = original_explanation_distance / distance_to_explain
        shap_values_transformed = shap_values / distance_coefficient
        return shap_values_transformed, expected_value_transformed

    # @st.cache
    # def shap_plot(model,X_tr,id):
    #     explainer = shap.TreeExplainer(model)
    #     shap_values_Model = explainer.shap_values(X_tr)
    #     fig = shap.force_plot(explainerModel.expected_value, shap_values_Model[id], S.iloc[[id]])
    #     return fig

    #################################
    #################################
    # Configuration of the streamlit page
    st.set_page_config(page_title='Loan application scoring dashboard',
                       page_icon='random',
                       layout='centered',
                       initial_sidebar_state='auto')

    # Display the title
    st.title('Loan application scoring dashboard')
    st.header("Prédictions de scoring client et comparaison à l'ensemble des clients")
    st.subheader("Laien WU - Data Science project 7")

    # Display the logo in the sidebar
    st.sidebar.image('logo.svg', width=180)

    error_flag = 0
    # df = pd.read_pickle('./data/shorted_data.pickle')
    filename = file_selector('./data/')
    df = pd.read_csv(filename)
    st.write(df.columns)
    clf = get_model(URL + '/models/clf_model.pickle')

    # SK_IDS = ['Overview'] + random.sample(get_sk_id_list(df), 20)
    SK_IDS = ['Overview'] + get_sk_id_list(df)
    df_short = df[df['SK_ID_CURR'].isin(get_sk_id_list(df))]
    X_train = df_short.drop(columns=['TARGET', "SK_ID_CURR"])
    y_train = df_short['TARGET']

    select_sk_id = st.sidebar.selectbox("Select SK_ID from list for a show : ", SK_IDS)

    if select_sk_id == 'Overview':
        pass
        if st.sidebar.checkbox("Features importances", key="29"):
            tmp = pd.get_dummies(X_train)
            model = xgboost.XGBClassifier().fit(tmp, y_train)
            explainer = shap.Explainer(model, tmp)
            shap_values = explainer(tmp)

            with st.spinner('Plot creation in progress...'):
                st.header("Graphique de l'importance des features")

                fig, axes = plt.subplots()
                shap.summary_plot(shap_values, tmp, plot_type='bar')
                st.pyplot(fig)

                st.header("Shape values des features")
                fig, axes = plt.subplots()
                # compute SHAP values
                explainer = shap.Explainer(model, tmp)
                shap_values = explainer(tmp)
                shap.summary_plot(shap_values, tmp)
                st.pyplot(fig)

        if st.sidebar.checkbox('Features description', key="a"):
            st.markdown("## Features descriptions")
            tmp = get_features_description()
            feature = st.selectbox("list of features: ,", tmp['Features'].tolist(), key="11")
            st.table(tmp[tmp['Features'] == feature])

            if st.checkbox('show full list', key="12"):
                st.table(tmp)
        else:
            df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
            st.markdown('### AGE: ')
            st.markdown("L'âge du client")
            fig, axes = plt.subplots()
            sns.kdeplot(df.loc[df['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label='No default risks')
            sns.kdeplot(df.loc[df['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label='Default risks')
            plt.xlabel('Age (années)')
            plt.ylabel('Densité')
            plt.title('Distribution des âges')
            plt.legend()
            st.pyplot(fig)

            if st.checkbox('show analysis', key="0"):
                st.markdown('#### Analysis')
                st.markdown('* Client having age  less than 40 have high probability of being default. ')
                st.markdown('* Client having age  greater than 40 have high probability of repaying loan.')
                st.markdown('* There is a visible sepration between two classes.')
                st.markdown('#### Conclusion')
                st.markdown('Younger clients are more likely to default as compared to older.')

            plt.figure(figsize=(8, 8))

            st.markdown("### Défaut de paiement des prêts selon les catégories d'âges")

            age_data = df[['TARGET', 'DAYS_BIRTH']]
            age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
            age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))
            age_groups = age_data.groupby('YEARS_BINNED').mean()

            fig, axes = plt.subplots()

            plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

            # Plot labeling
            plt.xticks(rotation=75)
            plt.xlabel('Age Group (years)')
            plt.ylabel('Failure to Repay (%)')
            plt.title('Failure to Repay by Age Group')
            st.pyplot(fig)

            if st.checkbox('show analysis', key="1"):
                st.markdown('#### Analysis')
                st.markdown(
                    "Le graphique montrent un risque bien supérieur chez les demandeurs de prêt dans les catégories d'âge jeunes.")

            st.markdown('### EXT_SOURCE_3: ')
            st.markdown('Normalized score from external data source.')
            fig, axes = plt.subplots()
            axes.set_xlabel('EXT_SOURCE_3')
            axes.set_ylabel('Density')
            sns.kdeplot(df.loc[df['TARGET'] == 0, 'EXT_SOURCE_3'], label='No default risks')
            sns.kdeplot(df.loc[df['TARGET'] == 1, 'EXT_SOURCE_3'], label='Default risks')
            plt.legend()
            st.pyplot(fig)

            if st.checkbox('show analysis', key="2"):
                st.markdown('#### Analysis')
                st.markdown('* External source 3 < 0.4 indicate high probability that client Default risks loan.')
                st.markdown(
                    '* (External source 3 > 0.5 and External source 3 < 0.9) indicate high probability that client No default risks loan.')
                st.markdown('* There is a visible sepration between two classes.')

                st.markdown('#### Conclusion')
                st.markdown('* External source 3 is a useful feature.')

            st.markdown('### AMT_ANNUITY: ')
            st.markdown(
                'Annuities are basically loans that are paid back over a set period of time at a set interest rate with consistent payments each period.')
            fig, axes = plt.subplots()
            axes.set_xlabel('AMT_ANNUITY')
            axes.set_ylabel('Density')
            sns.kdeplot(df.loc[(df['TARGET'] == 0) & (df['AMT_ANNUITY'] < (100000)), 'AMT_ANNUITY'],
                        label='No default risks')
            sns.kdeplot(df.loc[(df['TARGET'] == 1) & (df['AMT_ANNUITY'] < (100000)), 'AMT_ANNUITY'],
                        label='Default risks')
            plt.legend()
            st.pyplot(fig)

            if st.checkbox('show analysis', key="3"):
                st.markdown('#### Analysis')
                st.markdown('* Amount less than 10000 there is more chance that client No default risks.')
                st.markdown(
                    '* Amount between 20000 to 40000 shows a slight high probability that client Default risks loan.')
                st.markdown(
                    '* Amount greater than 40000 but less than 80000 shows a slight high probability that loan will be repayed.')
                st.markdown('* There is a visible sepration between two classes.')

                st.markdown('#### Conclusion')
                st.markdown('* Amount Annuity is a useful feature.')

            st.markdown("### D'autres graphiques de description générale")

            app_train_domain = df.copy()

            app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain[
                'AMT_INCOME_TOTAL']
            app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain[
                'AMT_INCOME_TOTAL']
            app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
            app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain[
                'DAYS_BIRTH']

            plt.figure(figsize=(6, 20))
            # iterate through the new features
            for i, feature in enumerate(
                    ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
                # create a new subplot for each source
                plt.subplot(4, 1, i + 1)
                # plot repaid loans
                sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature])
                # plot loans that were not repaid
                sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature])

                # Label the plots
                plt.title('Distribution of %s by Target Value' % feature)
                plt.xlabel('%s' % feature)
                plt.ylabel('Density')

            plt.tight_layout(h_pad=2.5)
            st.pyplot(plt)

    else:

        st.write('You selected: ', select_sk_id)

        st.markdown("#### Les données du client")

        st.table(df_short[df_short['SK_ID_CURR'] == select_sk_id])
        tmp = df_short[df_short['SK_ID_CURR'] == select_sk_id]
        X_data = tmp.drop(columns=['TARGET', 'SK_ID_CURR'])
        X_idx = df_short.index.get_loc(tmp.index[0])
        default_prob = round(clf.predict_proba(X_data)[:, 1][0] * 100, 2)

        st.write('Ce client présente une probabilité de défaut de paiement de: ', default_prob, '%')

        if default_prob < 50:
            st.markdown("## **Accord du prêt possible**")
        else:
            st.markdown("## **Ne pas accorder le prêt**")

        if st.checkbox("Détails d'analyse", key="22"):
            preproc_step = clf.named_steps['col_trans']
            featsel_step = clf.named_steps['feat_select']
            clf_step = clf.named_steps['reg']

            X_tr_prepro = preproc_step.transform(X_train)
            X_tr_prepro_df = pd.DataFrame(X_tr_prepro,
                                          index=X_train.index,
                                          columns=get_ct_feature_names(preproc_step))
            preproc_cols = X_tr_prepro_df.columns
            featsel_cols = preproc_cols[featsel_step.get_support()]
            X_tr_featsel = X_tr_prepro_df[featsel_cols]

            clf_step.fit(X_tr_featsel, y_train)

            features_sel = X_tr_featsel
            target_sel = y_train

            features_cust = features_sel.iloc[X_idx: X_idx + 1]
            target_cust = y_train.iloc[X_idx: X_idx + 1]

            neigh = NearestNeighbors(n_neighbors=20)
            neigh.fit(features_sel)

            nearest_cust_idx = neigh.kneighbors(X=features_cust,
                                                n_neighbors=20,
                                                return_distance=False).ravel()

            # features of neighbors
            features_neigh = features_sel.iloc[nearest_cust_idx]
            target_neigh = y_train.iloc[nearest_cust_idx]
            # features of customers and neighbors
            features_neigh_ = pd.concat([features_neigh, features_cust], axis=0)
            target_neigh_ = y_train.loc[features_neigh_.index]

            y_all = y_train.replace({0: 'repaid (global)',
                                     1: 'not repaid (global)'})
            X_neigh = X_tr_featsel.iloc[nearest_cust_idx]
            y_neigh = y_train.iloc[nearest_cust_idx].replace({0: 'repaid (neighbors)',
                                                              1: 'not repaid (neighbors)'})
            X_cust = X_tr_featsel.iloc[X_idx: X_idx + 1]
            main_cols = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'DAYS_BIRTH', 'EXT_SOURCE_1', 'CODE_GENDER_F',
                         'AMT_ANNUITY']

            # st.header("Boxplot des principaux features et leur dispersions")
            # fig, axes = plt.subplots()
            # plot_boxplot_var_by_target(X_tr_featsel, y_all, X_neigh, y_neigh, X_cust,
            #                            main_cols, figsize=(15, 4))
            # st.pyplot(fig)

            # st.header("Shap analyse (local)")
            #
            # explainer = shap.TreeExplainer(clf_step)
            # X_cust_neigh = pd.concat([X_neigh,
            #                           X_cust.to_frame(customer_idx).T],
            #                          axis=0)
            #
            #
            # shap_val_neigh = explainer.shap_values(X_cust_neigh)
            # expected_value = explainer.expected_value[1]
            #
            # shap_values = shap_val_neigh
            #
            # # vals= np.abs(shap_values).mean(0)
            # vals = np.abs(shap_values[1]).mean(0)
            #
            # feat_imp = pd.DataFrame(list(zip(X_cust_neigh.columns, vals)),
            #                         columns=['col_name', 'feature_imp']) \
            #     .sort_values(by=['feature_imp'], ascending=False)
            #
            # most_imp_10_cols = feat_imp.iloc[:10]['col_name'].values
            # shap_values_trans, expected_value_trans = \
            #     shap_transform_scale(shap_values=explainer.shap_values(X_cust_neigh)[1][-1],
            #                          expected_value=explainer.expected_value[1],
            #                          model_prediction=clf_step.predict_proba(X_cust_neigh)[:, 1][-1])
            # shap.plots._waterfall.waterfall_legacy(expected_value_trans,  # expected_value,
            #                                        shap_values_trans,  # shap_values[1][-1],
            #                                        X_cust_neigh.values.reshape(-1),
            #                                        feature_names=X_neigh.columns,
            #                                        max_display=10, show=False)
            # plt.gcf().set_size_inches((14, 6))
            # plt.show()
            explainerModel = shap.TreeExplainer(clf_step, X_tr_featsel)
            shap_values_Model = explainerModel.shap_values(X_tr_featsel)
            st.write("##", X_idx)
            # fig, axes = plt.subplots()

            # shap.force_plot(explainerModel.expected_value, shap_values_Model[X_idx], X_tr_featsel.iloc[[X_idx]])
            # # shap.force_plot(explainer.expected_value, shap_values(X_idx), X_tr_featsel.iloc[[X_idx]])
            # st.pyplot(fig)

            fig, axes = plt.subplots()
            shap.plots.waterfall(shap_values_Model[X_idx])
            st.pyplot(fig)


if __name__ == '__main__':
    main()
