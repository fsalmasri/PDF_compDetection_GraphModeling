
def adjust_bbx_margin(bbx, margin):
    x0, y0, xn, yn = bbx
    width = xn - x0
    height = yn - y0



    diff_w = ((width * (1 + margin)) - width) / 2
    diff_h = ((height * (1 + margin)) - height) / 2

    diff = min(diff_h, diff_w)

    x0 -= diff
    y0 -= diff
    xn += diff
    yn += diff

    return tuple([x0, y0, xn, yn])
