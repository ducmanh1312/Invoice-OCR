import re



def regex(text : str) -> bool:
    time_corpus = ['Ngày','NGÀY', 'GIỜ', 'Giờ', 'Gio', 'THỜI GIAN', 'Thời gian','PRINT TIME', 'Print time', 'NGÀY KINH DOANH', 'Date', 'DATE', 'Order']
    date_pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b'
    time_pattern = r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b|\b([01]?\d|2[0-3])\.[0-5]\d\b'
    combined_pattern = f'({date_pattern})|({time_pattern})'
    matches = re.findall(combined_pattern, text)
    if matches:
        return True
    else:
        for corpus in time_corpus:
            match = re.search(r'\b' + re.escape(corpus) + r'\b', text)
            if match:
                return True
        return False


# if __name__ == '__main__':
#     print(regex('Ngày: 16/08/2020 - 09:39'))