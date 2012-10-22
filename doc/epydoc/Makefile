PKGNAME = diffpy.srreal
OUTPUT = html
WEBDOCDIR = login.cacr.caltech.edu:/cacr/home/proj/danse/docs.danse.us/docroot/diffraction
WEBNAME = $(PKGNAME)-0.2


all:
	epydoc --config=epydoc.cfg --output=$(OUTPUT) \
	       --name=$(PKGNAME) $(PKGNAME)


clean:
	/bin/rm -rf $(OUTPUT)


publish:
	rsync -e /usr/bin/ssh -va --chmod=a+rwx,o-w,Dg+s,Fa-x \
	    $(OUTPUT)/ $(WEBDOCDIR)/$(WEBNAME)/
