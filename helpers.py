
# For the FeatureUnion Pipelines, because each step of it has to
# implement both the fit and transform methods. See the 'steps'
# argument here:
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
class ItemSelector():
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        text_columns = [
            'description',
            'company_profile',
            'requirements',
            'benefits',
            'title',
            'dep_ind_fun',
            'location',
            'exp_edu',
            'employment_type'
        ]

        numeric_columns = [
            'telecommuting',
            'has_company_logo',
            'has_questions',
            'min_salary_factor',
            'max_salary_factor',
            'benefits_len',
        ]

        if self.key in text_columns:
            return data[self.key].astype(str)
        elif self.key in numeric_columns:
            # import pdb
            # pdb.set_trace()
            return data[self.key].values.reshape(-1,1)

# Dictionaries for the transformation of the salary_range to
# min_salary factor and max_salary_factor
# Country to Currency mapping
currency = {
    'AE': 'AED', # United Ara Emirated Dirham
    'AM': 'AMD', # Armenian Dram
    'AT': 'EUR', # Euro
    'AU': 'AUD', # Australian Dollar
    'BE': 'EUR', # Euro
    'BG': 'BGN', # Bulgarian Leva
    'BR': 'BRL', # Brazilian Real
    'CA': 'CAD', # Canadian Dollar
    'CH': 'CHF', # Swiss Franc
    'DE': 'EUR', # Euro
    'DK': 'DKK', # Danish Krone
    'EE': 'EUR', # Euro
    'EG': 'EGP', # Egyptian Pound
    'ES': 'EUR', # Euro
    'FI': 'EUR', # Euro
    'FR': 'EUR', # Euro
    'GB': 'GBP', # Pound Sterling
    'GR': 'EUR', # Euro
    'HK': 'HKD', # Hong Kong Dollar
    'ID': 'IDR', # Indonesian Rupiah
    'IE': 'EUR', # Euro
    'IL': 'ILS', # Israeli Shekel
    'IN': 'INR', # Indian Rupee
    'IQ': 'IQD', # Iraqi Dinar
    'IS': 'ISK', # Icelandic Krona
    'IT': 'EUR', # Euro
    'JP': 'JPY', # Japanese Yen
    'KH': 'KHR', # Cambodian Riel
    'KW': 'KWD', # Kuwaiti Dinar
    'KZ': 'KZT', # Kazakhstani Tenge
    'MA': 'MAD', # Moroccan Dirham
    'MT': 'EUR', # Euro
    'MU': 'MUR', # Mauritian Rupee
    'MY': 'MYR', # Malaysian Ringgit
    'NG': 'NGN', # Nigerian Naira
    'NL': 'EUR', # Euro
    'NZ': 'NZD', # New Zealand Dollar
    'PH': 'PHP', # Philippine Peso
    'PK': 'PKR', # Pakistani Rupee
    'PL': 'PLN', # Polish Zloty
    'PT': 'EUR', # Euro
    'QA': 'QAR', # Qatari Rial
    'RO': 'RON', # Romanian Leu
    'RU': 'RUB', # Russian Rubble
    'SA': 'ZAR', # South African Rand
    'SE': 'SEK', # Swedish Krona
    'SG': 'SGD', # Singapore Dollar
    'TH': 'THB', # Thai Baht
    'TR': 'TRY', # Turkish Lira
    'TT': 'TTD', # Trinidad and Tobago Dollar
    'TW': 'TWD', # New Taiwan Dollar
    'ZA': 'ZAR', # South African Rand
    'UA': 'UAH', # Ukrainian Hryvnia
    'US': 'USD', # United States Dollar
}

# Any Currency to USD - The value it has to be multiplied with to be expressed in USD
# Rates as of May 6, 2020
# These rates could be obtained through various means, like for example through the
# Yahoo Finance libraries.
exchange_rate_USD = {
    'AED': 0.27,
    'AMD': 0.0021,
    'AUD': 0.64,
    'BGN': 0.56,
    'BRL': 0.18,
    'CAD': 0.71,
    'CHF': 1.03,
    'DKK': 0.15,
    'EGP': 0.064,
    'EUR': 1.08,
    'GBP': 1.24,
    'HKD': 0.13,
    'IDR': 0.000067,
    'ILS': 0.28,
    'INR': 0.013,
    'IQD': 0.00084,
    'ISK': 0.007,
    'JPY': 0.0093,
    'KHR': 0.00024,
    'KWD': 3.24,
    'KZT': 0.0024,
    'MAD': 0.10,
    'MUR': 0.025,
    'MYR': 0.23,
    'NGN': 0.0026,
    'NZD': 0.61,
    'PHP': 0.020,
    'PKR': 0.0063,
    'PLN': 0.24,
    'QAR': 0.27,
    'RON': 0.23,
    'RUB': 0.014,
    'ZAR': 0.054,
    'SEK': 0.10,
    'SGD': 0.71,
    'THB': 0.031,
    'TRY': 0.14,
    'TTD': 0.15,
    'TWD': 0.034,
    'UAH': 0.037,
    'USD': 1.0,
    'ZAR': 0.057,
}

# Minimum salary of a country expressed in USD
# Data as of May 6 - 2020 -- https://en.wikipedia.org/wiki/List_of_minimum_wages_by_country
minimum_salary_USD = {
    'AE': 9801, # For no high school diploma
    'AM': 1690,
    'AT': 19628,
    'AU': 28768,
    'BE': 22565,
    'BG': 4399,
    'BR': 3500,
    'CA': 18112,
    'CH': 26940, # Approximation
    'DE': 22525,
    'DK': 2580, # Approximation - https://checkinprice.com/average-minimum-salary-copenhagen-denmark/
    'EE': 7877,
    'EG': 816, # 68 per month for the public sector - approximation
    'ES': 15687,
    'FI': 31200,
    'FR': 21142,
    'GB': 24183, # differs by age - approximation
    'GR': 10731,
    'HK': 10049,
    'ID': 1304,
    'IE': 24166,
    'IL': 17667,
    'IN': 767,
    'IQ': 2534,
    'IS': 28077,
    'IT': 11000, # approximation - No minimum salary in Italy, mostly collective agreements.
    'JP': 14881,
    'KH': 2280, # approximation
    'KW': 2400,
    'KZ': 1041,
    'MA': 3709,
    'MT': 11019,
    'MU': 1513, # approximation
    'MY': 2566,
    'NG': 1179,
    'NL': 22876,
    'NZ': 27881,
    'PH': 2770,
    'PK': 1991,
    'PL': 8254,
    'PT': 9910,
    'QA': 2471, # https://tradingeconomics.com/qatar/minimum-wages
    'RO': 6163,
    'RU': 2495,
    'SA': 3511,
    'SE': 27840, # approximation
    'SG': 9000, # approximation - no minimum wage
    'TH': 3034,
    'TR': 9676,
    'TT': 4602,
    'TW': 9088,
    'UA': 2131,
    'US': 15080,
    'ZA': 3511,
}
