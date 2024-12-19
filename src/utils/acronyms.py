GERMAN_ACRONYMS = {
    # Deutsche Politik
    "SPD", "CDU", "CSU", "FDP", "AFD", "DIE LINKE", "SSW", "NPD",
    "GRÜNE", "PIRATEN", "DVU", "PDS",
    
    # Deutsche Behörden & Institutionen
    "BND", "BKA", "LKA", "GEZ", "KMK", "BSI", "BaFin", "DFG",
    "DAAD", "GIZ", "KfW", "THW", "BSI", "BAföG",
    
    # Deutsche Medien
    "ARD", "ZDF", "WDR", "NDR", "BR", "HR", "MDR", "RBB", "SR", "SWR",
    "RTL", "SAT1", "PRO7", "VOX", "KIKA", "ARTE", "DPA", "FAZ",
    
    # Deutsche Wirtschaft
    "DAX", "BMW", "VW", "AUDI", "DHL", "SAP", "RWE", "EON", "BASF",
    "DM", "ALDI", "LIDL", "EDEKA", "REWE", "TUI", "BVB", "HDI",
    "HUK", "LBS", "ING", "KSK", "DBK",
    
    # Deutsche Verbände & Organisationen
    "DGB", "IG", "VDI", "ADAC", "DLRG", "DRK", "AWO", "VDA",
    "DIHK", "BDI", "GEW", "VDE", "TÜV", "DEKRA",
    
    # Bildung & Forschung
    "TUM", "LMU", "KIT", "RWTH", "MPG", "FHG", "HU", "FU",
    "TU", "HAW", "DLR", "GSI", "HPI",
    
    # Sport
    "DFB", "DFL", "FCB", "BVB", "HSV", "VFB", "FSV", "TSG",
    "UEFA", "FIFA", "IOC", "DSB", "DOSB",
    
    # Internationale Politik & Organisationen
    "USA", "NATO", "UN", "EU", "EZB", "WHO", "UNESCO", "UNHCR",
    "UNICEF", "WTO", "IWF", "OPEC", "IAEA", "OECD", "UNO",
    "FBI", "CIA", "MI6", "KGB", "RAF", "ISIS",
    
    # Internationale Wirtschaft
    "IBM", "HP", "AMD", "NASA", "HSBC", "UBS", "BBC", "CNN",
    "MSFT", "AAPL", "GOOG", "AMZN", "TSLA", "NVDA", "META",
    "IKEA", "VISA", "UBER", "FIFA", "NYSE", "FTSE",
    
    # Technologie
    "USB", "WLAN", "GPS", "LED", "LCD", "CPU", "RAM", "SSD",
    "DNS", "HTML", "CSS", "PHP", "SQL", "API", "SDK", "LTE",
    "UMTS", "ISDN", "DSL", "IPv4", "IPv6", "HTTP", "FTP",
    
    # Rechtliches & Geschäftliches
    "GmbH", "AG", "KG", "OHG", "EG", "MBA", "CEO", "CFO",
    "CTO", "COO", "GbR", "UG", "SE", "KGaA",
    
    # Sonstiges
    "PDF", "ISBN", "IBAN", "BIC", "PIN", "TAN", "SMS", "MMS",
    "DVD", "CD", "TV", "PC", "ID", "FAQ", "PKW", "LKW"
}

# Kategorisierte Version
CATEGORIZED_ACRONYMS = {
    "Politik_DE": [
        "SPD", "CDU", "CSU", "FDP", "AFD", "DIE LINKE", "GRÜNE", "SSW"
    ],
    "Politik_International": [
        "USA", "NATO", "UN", "EU", "EZB", "WHO", "UNESCO", "UNHCR"
    ],
    "Medien_DE": [
        "ARD", "ZDF", "WDR", "NDR", "RTL", "PRO7", "SAT1", "VOX"
    ],
    "Wirtschaft_DE": [
        "BMW", "VW", "SAP", "RWE", "EON", "BASF", "ALDI", "LIDL"
    ],
    "Wirtschaft_International": [
        "IBM", "HP", "MSFT", "AAPL", "GOOG", "AMZN", "TSLA", "META"
    ],
    "Sport": [
        "DFB", "DFL", "FCB", "BVB", "HSV", "UEFA", "FIFA", "IOC"
    ],
    "Technologie": [
        "USB", "WLAN", "GPS", "LED", "CPU", "RAM", "HTML", "SQL"
    ],
    "Behörden": [
        "BND", "BKA", "LKA", "BSI", "GEZ", "KMK", "DFG", "THW"
    ]
}

def is_valid_acronym(model, acronym):
    """
    Prüft, ob ein Akronym im FastText-Modell vorhanden ist
    """
    try:
        _ = model.get_word_vector(acronym)
        return True
    except:
        return False