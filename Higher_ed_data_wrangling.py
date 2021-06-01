import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def loadStatic():
    '''a function to load static higher education provider data'''
    
    df = pd.read_csv("Provider.csv").fillna(
        value={'affiliation': 'NUHEP', 'affil_status': 'current'})
    
    return df




def loadEftsl():
    '''a function to load equivalent full-time student load (eftsl)'''
    
    df = pd.read_csv('19_381_Onshore_EFTSL.csv')
    
    return df




def loadPerformance():
    
    '''a function to load, clean and join student success rate and attrition rate data'''
    
    #load success rate data
    success_all = pd.read_csv("19_381_Onshore_success_rate_overall.csv").rename(
        columns = {"success_rate": "success_all"}).drop(['provider_name'], axis=1)
    
    success_dom = pd.read_csv("19_381_Onshore_success_domestic.csv").rename(
        columns = {"success_rate": "success_dom"})[['code_year', 'success_dom']]
    
    success_int = pd.read_csv("19_381_Onshore_success_international.csv").rename(
        columns = {"success_rate": "success_int"})[['code_year', 'success_int']]
    
    # load attrition rate data
    attrition_all = pd.read_csv("19_381_Onshore_attrition_rate_overall.csv").rename(
        columns = {'attrition_rate': 'attrition_all'})[['code_year', 'attrition_all']]
    
    attrition_dom = pd.read_csv("19_381_Onshore_attrition_domestic.csv").rename(
        columns = {'attrition_rate': 'attrition_dom'})[['code_year', 'attrition_dom']]
    
    attrition_int = pd.read_csv("19_381_Onshore_attrition_international.csv").rename(
        columns = {'attrition_rate': 'attrition_int'})[['code_year', 'attrition_int']]
    
    # merge data frames
    df = success_all.merge(success_dom, on='code_year', how = 'outer').merge(
        success_int, on='code_year', how = 'outer').merge(
        attrition_all, on='code_year', how = 'outer').merge(
        attrition_dom, on='code_year', how = 'outer').merge(
        attrition_int, on='code_year', how = 'outer')
    
    return df[df['ref_year'] > 2012]




def loadStaff(eftsl):

    '''a function to load staff data and create important staff-related features'''
    
    # load .csv
    staff = pd.read_csv('19_381_Academic_staff.csv').rename(
        columns = {'tab1_academic_FTE': 'all_fte', 
                   'tab2_academic_FTE': 'salaried_fte', 
                   'tab3_senior_FTE': 'senior_fte', 
                   'tab3_senior_headcount_mod': 'senior_headcount'})
    
    # calculate pure staffing features
    staff['sessional_fte'] = staff['all_fte'] - staff['salaried_fte']
    staff['sessional_prop'] = staff['sessional_fte'] / staff['all_fte']
    staff['senior_prop'] = staff['senior_fte'] / staff['all_fte']
    
    # join with eftsl data to enable calculation of student:staff ratio
    eftsl_total = eftsl[['code_year', 'EFTSL']].groupby(['code_year']).sum().reset_index()
    staff = staff.merge(eftsl_total, on='code_year').rename(columns={'EFTSL': 'eftsl'})
    
    # calculate student:staff ratios
    staff['ssr_all'] = staff['eftsl'] / staff['all_fte']
    staff['ssr_salaried'] = staff['eftsl'] / staff['salaried_fte']
    
    # select staffing features
    df = staff[['code_year', 
                'all_fte', 
                'salaried_fte', 
                'senior_fte', 
                'senior_headcount', 
                'sessional_fte', 
                'sessional_prop', 
                'senior_prop', 
                'ssr_all']]

    return df




def buildPostgrad(eftsl):
    
    ''' a function to extract postgraduate student proportions from eftsl data'''
    
    # aggregate eftsl at course level by provider by year
    df = eftsl.drop(['provider_code', 'ref_year'], axis=1).groupby(['Course_level','code_year']).sum().reset_index()
    
    # pivot in order to make columnwise calculation
    df = df.pivot(index='code_year', columns='Course_level', values='EFTSL').fillna(value=0)
    
    # flatten index after pivoting
    df.columns = [''.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace = True)
    
    # calculate postgraduate eftsl proportion and drop unnecessary columns
    df['postgrad_prop'] = df['Postgrad by course'] / (df['Postgrad by course'] + df['Undergrad'])
    df = df.drop(['Postgrad by course', 'Undergrad'], axis=1).fillna(value=0)
    
    return df




def buildInternational(eftsl):
    
    ''' a function to extract international student proportions from eftsl data'''
    
    # aggregate eftsl at citizenship type (domestic or international) by provider by year
    df = eftsl.drop(['provider_code', 'ref_year'], axis=1).groupby(['citizenship','code_year']).sum().reset_index()
    
    # pivot in order to make columnwise calculation
    df = df.pivot(index='code_year', columns='citizenship', values='EFTSL').fillna(value=0)
    
    # flatten index after pivoting
    df.columns = [''.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace = True)
    
    # calculate international eftsl proportion and drop unnecessary columns
    df['int_prop'] = df['International'] / (df['International'] + df['Domestic'])
    df = df.drop(['International', 'Domestic'], axis=1).fillna(value=0)
    
    return df




def buildBfoe(eftsl):
    
    '''a function to generate eftsl and bfoe-related features'''

    # aggregate eftsl at BFOE type by provider by year
    df = eftsl.drop(['provider_code', 'ref_year'], axis=1).groupby(['primary_BFOE','code_year']).sum().reset_index()

    # pivot wider and replace NAs with zeros to ensure each provider in each year has a value for each BFOE
    df = df.pivot(index='code_year', columns='primary_BFOE', values='EFTSL').fillna(value=0)

    # rename BFOE categories to be shorter
    df = df.rename(columns = {'01 Natural and Physical Sciences': 'nat_phys_sci',
                             '02 Information Technology': 'info_tech',
                             '03 Engineering and Related Technologies': 'engineering',
                             '04 Architecture and Building': 'arch_build',
                             '05 Agriculture, Environmental and Related Studies': 'agri_env',
                             '06 Health': 'health',
                             '07 Education': 'education',
                             '08 Management and Commerce': 'mge_com',
                             '09 Society and Culture': 'soc_cult',
                             '10 Creative Arts': 'creat_art',
                             '11 Food, Hospitality and Personal Services': 'food_hosp',
                             '12 Mixed Field Programmes': 'mixed',
                             '13 Non-award courses': 'non_award'})

    vars = ['nat_phys_sci', 
            'info_tech', 
            'engineering', 
            'arch_build', 
            'agri_env', 
            'health', 
            'education', 
            'mge_com', 
            'soc_cult', 
            'creat_art', 
            'food_hosp',
            'mixed',
            'non_award']

    # flatten index after pivoting
    df.columns = [''.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace = True)

    # return to long format to implement column-wise calculations
    df = pd.melt(df, id_vars=['code_year'], value_vars=vars, var_name='bfoe', value_name='eftsl')

    # find max bfoe eftsl for each provider
    df['bfoe_max'] = df.groupby(['code_year'])['eftsl'].transform(max)
    df['bfoe_max'].replace(0, np.nan, inplace=True)

    # caclulate total eftsl for ech provider
    df['eftsl_sum'] = df.groupby(['code_year'])['eftsl'].transform(sum)

    # calculate eftsl proportion for each bfoe for each provider for each year
    df['eftsl_prop'] = df['eftsl'] / df['eftsl_sum']
    df['eftsl_prop'].replace(0, np.nan, inplace=True)

    # calculate values to be summed for entropy calculation
    df['pre_entropy'] = -1*df['eftsl_prop']*np.log2(df['eftsl_prop'])

    #calculate entropy
    df['bfoe_entropy'] = df.groupby(['code_year'])['pre_entropy'].transform(sum)

    # calculate values to be summed for gini impurity calculation
    df['pre_gini'] = df['eftsl_prop']*(1-df['eftsl_prop'])

    # calculate gini impurity
    df['bfoe_gini_impurity'] = df.groupby(['code_year'])['pre_gini'].transform(sum)

    #filter to only the max bfoe for any given year
    df = df[df['bfoe_max'] == df['eftsl']] 
    df = df[df['bfoe_max'] != 0]

    # drop unnecessary columns and rename as required
    df = df.drop(['eftsl', 'bfoe_max', 'pre_entropy', 'pre_gini'], axis=1).rename(
        columns={'bfoe': 'primary_bfoe', 'eftsl_prop': 'primary_bfoe_prop'})

    return df




def loadAverages():
    
    '''a function to load subsector averages published by TEQSA'''
    
    TEQSA_avg = pd.read_csv('TEQSA_averages.csv')
    
    return TEQSA_avg




def loadDecisions():
    decisions = pd.read_csv("https://data.gov.au/data/dataset/0c4f6591-2aea-4797-a127-ae8f8a0be0e2/resource/a61abca4-a4b7-4f17-a975-0bba20c2c73f/download/outcomes_26102020.csv")
    decisions = decisions.drop('Date',axis=1)
    providers = pd.read_csv('https://data.gov.au/data/dataset/0c4f6591-2aea-4797-a127-ae8f8a0be0e2/resource/07370e3f-780b-4a70-8c87-b6796d5ab237/download/providers_21102020.csv')
    providers = providers[['ProviderID', 'ProviderName', 'Category']]

    decisions['Description'] = decisions.Title.str.rsplit(n=3).str[0]
    decisions['Year'] = decisions.Title.str.rsplit(n=1).str[-1]

    decisions = decisions.drop('Title',axis=1)

    decisions['dec_condition'] = decisions['Text'].str.contains('ondition')|decisions['Description'].str.contains('ondition')
    decisions['dec_staff_ratio'] = decisions['Text'].str.contains('staff ratio|SSR')
    decisions['dec_sessional'] = decisions['Text'].str.contains('essional|asual')
    decisions['dec_attrition'] = decisions['Text'].str.contains('ttrition')
    decisions['dec_progress'] = decisions['Text'].str.contains('uccess|rogress')
    decisions['dec_rejected'] = decisions['Description'].str.contains('reject|Reject|not|Not|Cancel r|cancel r|Cancel a|cancel a')|decisions['CourseID'].str.contains('reject|Reject')
    decisions['extension'] = decisions['Description'].str.contains('Exten|exten')
    decisions['dec_negative'] = decisions['Text'].str.contains('ondition')|decisions['Description'].str.contains('ondition')|decisions['Description'].str.contains('reject|Reject|not|Not|Cancel r|cancel r|Cancel a|cancel a')|decisions['CourseID'].str.contains('reject|Reject')

    decisions = decisions[decisions['extension'] == False]

    decisions = decisions.drop('Text',axis=1)
    decisions['ProviderID'] = decisions['ProviderID'].str.strip()

    decisions = providers.merge(decisions, on = 'ProviderID', how = 'outer')

    decisions = decisions[decisions['Type'] == 'Decision']
    decisions = decisions.drop('Type',axis=1)
    decisions['DecisionType'] = decisions['DecisionType'].str.strip()

    decisions = decisions[decisions.DecisionType.isin(['Re-registration', 'Registration', 'Re-accreditation', 'Accreditation'])]

    decisions = decisions[[ 'ProviderID',
                            'Year', 
                            'dec_negative',
                            'dec_rejected',
                            'dec_condition',
                            'dec_staff_ratio',
                            'dec_sessional',
                            'dec_attrition',
                            'dec_progress']]
    decisions.rename(columns={"ProviderID" : "prv", "Year" : "ref_year"}, inplace=True)

    return decisions




def joinData():
    '''a function to join all higher ed data frames''' 

    # load datasets
    eftsl = loadEftsl()
    provider = loadStatic()
    performance = loadPerformance()
    staff = loadStaff(eftsl)
    teqsa_avg = loadAverages()
    decisions = loadDecisions()[['prv', 'ref_year', 'dec_negative']].dropna().drop_duplicates()
    decisions.ref_year = decisions.ref_year.astype(int)
    
    # transform eftsl data
    bfoe = buildBfoe(eftsl)
    postgrad = buildPostgrad(eftsl)
    international = buildInternational(eftsl)

    # join datasets
    df = provider.merge(performance, on='provider_code', how = 'outer').merge(
        staff, on='code_year', how = 'outer').merge(
        bfoe, on='code_year', how = 'outer').merge(
        postgrad, on='code_year', how = 'outer').merge(
        international, on='code_year', how = 'outer').merge(
        teqsa_avg, on=['TEQSA_type', 'ref_year'], how = 'outer').merge(
        decisions, on=['prv', 'ref_year'], how = 'left')
    # remove providers without a name
    df = df[df.provider_name.notnull()]

    # remove ref_year before 2012
    df = df[df['ref_year'] > 2011]

    df.ref_year = df.ref_year.astype(int)

    # remove providers that no longer existed in the last year of the data collection (2017)
    still_exists = list(df[df['ref_year'] == 2017]['prv'])
    df = df[df.prv.isin(still_exists)]

    # replace unlikely values with null
    df['ssr_all'] = np.where((df.ssr_all < 0.00001), np.NaN ,df.ssr_all)
    df['sessional_prop'] = np.where((df.sessional_prop < 0.00001), np.NaN ,df.sessional_prop)
    df['sessional_prop'] = np.where((df.sessional_prop > 0.99999), np.NaN ,df.sessional_prop)


    # remove redundant provider code
    df = df.drop(['provider_code'], axis=1)
    
    # select and order columns
    

    return df.reset_index()




def impute(df):
    '''a function to impute missing values using MICE'''
    
    # select numeric features
    df2 = df[['primary_bfoe_prop',
                'bfoe_entropy',
                'bfoe_gini_impurity',
                'postgrad_prop',
                'int_prop',
                'sessional_prop_type_avg',
                'sessional_prop',
                'senior_prop',
                'ssr_all_type_avg',
                'ssr_salaried_type_avg',
                'ssr_all',
                'success_dom',
                'success_int',
                'success_all',
                'attrition_dom',
                'attrition_int',
                'attrition_all']]

    # specify minimum and maximum values for imputation
    min = [0.01, 0, 0, 0, 0, 0.01, 0.01, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    max = [1, 3.05, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    
    # instantiate imputer
    mice_imputer = IterativeImputer(max_iter=10000, random_state=1, min_value=min, max_value=max)
    
    # fit imputation and return as a dataframe
    df_imp = pd.DataFrame(mice_imputer.fit_transform(df2).round(decimals=6))
    
    # rename imputed columns
    df_imp.columns = ['primary_bfoe_prop',
                'bfoe_entropy',
                'bfoe_gini_impurity',
                'postgrad_prop',
                'int_prop',
                'sessional_prop_type_avg',
                'sessional_prop',
                'senior_prop',
                'ssr_all_type_avg',
                'ssr_salaried_type_avg',
                'ssr_all',
                'success_dom',
                'success_int',
                'success_all',
                'attrition_dom',
                'attrition_int',
                'attrition_all']
    
    # specify columns to be overwritten with imputed data, noting that cells without NAs will be untouched
    columns_to_overwrite = ['primary_bfoe_prop',
                'bfoe_entropy',
                'bfoe_gini_impurity',
                'postgrad_prop',
                'int_prop',
                'sessional_prop_type_avg',
                'sessional_prop',
                'senior_prop',
                'ssr_all_type_avg',
                'ssr_salaried_type_avg',
                'ssr_all',
                'success_dom',
                'success_int',
                'success_all',
                'attrition_dom',
                'attrition_int',
                'attrition_all']
    
    # drop columns to be overwritten from original dataset and replace with imputed data
    df3 = df.drop(columns_to_overwrite, axis=1)
    df3[columns_to_overwrite] = df_imp[columns_to_overwrite]

    return df3[df3['ssr_all'] >0]




def staffAdjust(df_imp, sessional_multiplier = 1, ssr_multiplier = 1):
    
    df_imp = df_imp
    
    '''a function to adjust imputed staffing values based on sub-sector averages published by TEQSA'''

    compare = df_imp[['ref_year','TEQSA_type',
                    'sessional_prop_type_avg',
                    'sessional_prop',
                    'ssr_all_type_avg',
                    'ssr_salaried_type_avg',
                    'ssr_all']]

    agg = compare.groupby(['ref_year','TEQSA_type']).mean().reset_index()
    agg['sessional_diff'] = agg['sessional_prop_type_avg'] - agg['sessional_prop']
    agg['ssr_all_diff'] = agg['ssr_all_type_avg'] - agg['ssr_all']
    diff = agg.drop(['sessional_prop_type_avg', 'sessional_prop', 'ssr_all_type_avg', 'ssr_all', 'ssr_salaried_type_avg'], axis=1)

    df_diff = df_imp.merge(diff, on=['ref_year', 'TEQSA_type'], how = 'left')

    df_diff.loc[df_diff.TEQSA_type!='university', 'sessional_prop'] = sessional_multiplier * (df_diff.loc[df_diff.TEQSA_type!='university', 'sessional_prop'] + df_diff.loc[df_diff.TEQSA_type!='university', 'sessional_diff'])
    df_diff.loc[df_diff.TEQSA_type!='university', 'ssr_all'] = ssr_multiplier * (df_diff.loc[df_diff.TEQSA_type!='university', 'ssr_all'] + df_diff.loc[df_diff.TEQSA_type!='university', 'ssr_all_diff'])

    df_diff = df_diff.drop(['sessional_diff', 'ssr_all_diff', 'sessional_prop_type_avg', 'ssr_all_type_avg', 'ssr_salaried_type_avg'], axis=1)

    df_diff['ssr_salaried'] = df_diff['ssr_all'] * (1 / (1 - df_diff['sessional_prop']))
    
    df = df_diff[['code_year',
            'prv',
            'provider_name',
            'ref_year',
            'type',
            'TEQSA_type',
            'ownership',
            'profit',
            'affiliation',
            'affil_status',
            'eftsl_sum',
            'primary_bfoe',
            'primary_bfoe_prop',
            'bfoe_entropy',
            'bfoe_gini_impurity',
            'postgrad_prop',
            'int_prop',
            'sessional_prop',
            'senior_prop',
            'ssr_all',
            'ssr_salaried',
            'success_dom',
            'success_int',
            'success_all',
            'attrition_dom',
            'attrition_int',
            'attrition_all',
            'dec_negative']]

    return df.round(decimals=3)




def main():
    df_raw = joinData()
    df_imp = impute(df_raw)
    df = staffAdjust(df_imp)
    df.to_csv('higher_ed_data.csv', index = False)
    return df




if __name__ == '__main__':
    main()