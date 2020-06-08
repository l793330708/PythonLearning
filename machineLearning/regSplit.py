import re
mySent = 'This book is the best book on Python or M.L. I have ever laid \
        eyes upon.'
regExp = re.compile('/w*')
arr = regExp.split(mySent)
print(arr)