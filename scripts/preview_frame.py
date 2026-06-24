import sys, os, re
sys.path.insert(0, '/app')
from PIL import Image, ImageDraw, ImageFont

def _find_font(size):
    for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

_EMOJI_FONT_PATHS = ['/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf']
_BADGE_COLORS = {
    '動物':(183,65,45),'食物':(155,120,45),'飲料':(49,79,113),
    '自然':(80,115,70),'交通':(90,75,110),'物品':(120,90,55),'身體':(160,85,90),
}

def _draw_emoji_on(base_img, emoji_str, x, y, size=260, tag=''):
    for ep in _EMOJI_FONT_PATHS:
        if not os.path.exists(ep): continue
        try:
            efont = ImageFont.truetype(ep, 109)
            canvas = Image.new('RGBA', (300,300),(0,0,0,0))
            cdraw = ImageDraw.Draw(canvas)
            cdraw.text((20,20), emoji_str, font=efont, embedded_color=True)
            bbox = canvas.getbbox()
            if bbox and (bbox[2]-bbox[0])>10:
                cropped=canvas.crop(bbox); ow,oh=cropped.size
                nw=max(1,int(ow*size/oh))
                canvas=cropped.resize((nw,size),Image.LANCZOS)
                base_img.paste(canvas,(x+(size-nw)//2,y),canvas); return
        except: pass
    color = _BADGE_COLORS.get(tag,(80,130,255))
    draw=ImageDraw.Draw(base_img); r=size//2; ccx,ccy=x+r,y+r
    draw.ellipse([(ccx-r,ccy-r),(ccx+r,ccy+r)],fill=color)
    label=tag[:2] if tag else (emoji_str[:1] if emoji_str else '?')
    draw.text((ccx,ccy),label,fill=(255,255,255),font=_find_font(r),anchor='mm')

def make_frame(word, out_path):
    W, H = 1920, 1080
    BG=(252,246,235); ORANGE=(218,108,48); CHARCOAL=(32,26,18)
    SLATE=(95,105,118); VERMIL=(183,65,45); WARM_GRAY=(135,123,108)

    img = Image.new('RGB',(W,H),BG)
    draw = ImageDraw.Draw(img)

    BORDER = 10
    draw.rectangle([(0,0),(W,BORDER)], fill=ORANGE)
    draw.rectangle([(0,H-BORDER),(W,H)], fill=ORANGE)
    draw.rectangle([(0,0),(BORDER,H)], fill=ORANGE)
    draw.rectangle([(W-BORDER,0),(W,H)], fill=ORANGE)

    jlpt = word.get('jlpt_level','')
    if jlpt:
        draw.rounded_rectangle([(W-152,22),(W-22,64)], radius=6, fill=ORANGE)
        draw.text((W-87,43), jlpt, fill=(252,246,235), font=_find_font(28), anchor='mm')

    EMOJI_SIZE = 260
    ex = W//2 - EMOJI_SIZE//2
    ey = H//2 - EMOJI_SIZE//2
    _draw_emoji_on(img, word.get('emoji',''), ex, ey, size=EMOJI_SIZE, tag=word.get('tags',''))

    kanji_str = word.get('kanji','')
    hira_str  = word.get('hiragana','')
    kata_str  = word.get('katakana','')
    def _is_kanji(c): return '一'<=c<='鿿' or '㐀'<=c<='䶿'
    def _is_kana(c):  return 'ぁ'<=c<='ヿ'
    _RUBY_OVERRIDE = {
        'でんわ':   [('電','でん'),('話','わ')],
        'ひこうき': [('飛','ひ'),('行','こう'),('機','き')],
        'ぼうし':   [('帽','ぼう'),('子','し')],
    }
    if kanji_str:
        kf = _find_font(148); rf = _find_font(58)
        if hira_str in _RUBY_OVERRIDE:
            assignments = _RUBY_OVERRIDE[hira_str]
        else:
            hlist = list(hira_str); klist = list(kanji_str)
            hi = 0; assignments = []
            for i, ch in enumerate(klist):
                if _is_kana(ch):
                    eq = chr(ord(ch)-0x60) if 'ァ'<=ch<='ヶ' else ch
                    while hi<len(hlist) and hlist[hi]!=eq: hi+=1
                    assignments.append((ch,''))
                    if hi<len(hlist): hi+=1
                else:
                    nxt = next((klist[j] for j in range(i+1,len(klist)) if _is_kana(klist[j])),None)
                    st=hi
                    if nxt:
                        eq=chr(ord(nxt)-0x60) if 'ァ'<=nxt<='ヶ' else nxt
                        while hi<len(hlist) and hlist[hi]!=eq: hi+=1
                    else:
                        rem_k=sum(1 for c in klist[i+1:] if _is_kanji(c))
                        hi=len(hlist) if rem_k==0 else hi+max(1,(len(hlist)-hi)//(rem_k+1))
                    assignments.append((ch,''.join(hlist[st:hi])))
        try: ws=[max(10,int(kf.getlength(ch))) for ch,_ in assignments]
        except AttributeError: ws=[148]*len(assignments)
        total_w=sum(ws); x=W//2-total_w//2
        KANJI_Y=232; RUBY_DY=148//2+8+58//2
        for (ch,ruby),w in zip(assignments,ws):
            ccx=x+w//2
            draw.text((ccx,KANJI_Y),ch,fill=CHARCOAL,font=kf,anchor='mm')
            if ruby: draw.text((ccx,KANJI_Y-RUBY_DY),ruby,fill=SLATE,font=rf,anchor='mm')
            x+=w
    else:
        draw.text((W//2,240), kata_str or hira_str, fill=CHARCOAL, font=_find_font(148), anchor='mm')

    draw.text((W//2,765), word.get('chinese_translation',''), fill=VERMIL, font=_find_font(82), anchor='mm')

    etym = word.get('etymology','')
    if etym:
        etym_clean = re.sub(r'\([A-Za-z ]+\)','',etym).strip()
        etym_clean = re.sub(r'（[^）]*）','',etym_clean).strip()
        etym_clean = re.sub(r'([^\s])是',r'\1 是',etym_clean)
        parts = [p.strip() for p in re.split(r'[，、]',etym_clean) if p.strip()]
        lines = ['；'.join(parts)] if len(parts)<=2 else parts[:3]
        f_e = _find_font(30)
        for li, line in enumerate(lines):
            if len(line)>48: line=line[:47]+'…'
            draw.text((W//2, 858+li*54), line, fill=WARM_GRAY, font=f_e, anchor='mm')

    img.save(out_path, 'PNG')
    print('saved', out_path)

samples = [
    {'hiragana':'ねこ','kanji':'猫','chinese_translation':'貓',
     'etymology':'','emoji':'🐱','tags':'動物','jlpt_level':'N5'},
    {'hiragana':'すし','kanji':'寿司','chinese_translation':'壽司',
     'etymology':'す（酢）是醋，し（飯）是飯，醋飯之意','emoji':'🍣','tags':'食物','jlpt_level':'N5'},
    {'hiragana':'でんしゃ','kanji':'電車','chinese_translation':'電車',
     'etymology':'でん（電）是電力，しゃ（車）是車，電力驅動的車','emoji':'🚃','tags':'交通','jlpt_level':'N5'},
    {'hiragana':'でんわ','kanji':'電話','chinese_translation':'電話',
     'etymology':'でん（電）是電力，わ（話）是對話，用電傳遞話語','emoji':'📞','tags':'物品','jlpt_level':'N5'},
    {'hiragana':'ひこうき','kanji':'飛行機','chinese_translation':'飛機',
     'etymology':'ひこう（飛行）是飛行，き（機）是機器','emoji':'✈️','tags':'交通','jlpt_level':'N5'},
    {'hiragana':'ぼうし','kanji':'帽子','chinese_translation':'帽子',
     'etymology':'ぼう（帽）是帽，し（子）是器物後綴，戴在頭上的東西','emoji':'🎩','tags':'物品','jlpt_level':'N5'},
]

os.makedirs('/app/videos', exist_ok=True)
for i, w in enumerate(samples):
    make_frame(w, f'/app/videos/prev_{i}.png')
