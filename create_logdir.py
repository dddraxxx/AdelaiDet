from pathlib import Path as pa

if __name__=='__main__':
    import sys
    path  = sys.argv[1]
    path = pa(path)
    for p in sorted(path.glob('events.out.tfevents*'))[:-1]:
        dir = pa(str(p)[-7:])
        pdir = (p.parent/dir)
        print(pdir)
        pdir.mkdir()
        p.rename(pdir/p.name)
