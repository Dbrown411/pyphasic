try:
    """figs showed downsampled signals using ulttb. default to no downsampling if not installed"""
    from ulttb import downsample
    ulttb = True
except:
    ulttb = False
try:
    from lttb import downsample
    lttb = True
except:
    lttb = False
