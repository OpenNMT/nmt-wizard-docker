#!/usr/bin/perl

my $type = shift;
my $file = @ARGV;


print '<?xml version="1.0" encoding="UTF-8"?>'."\n";
print '<!DOCTYPE mteval SYSTEM "ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.7.dtd">'."\n";
print '<mteval>'."\n";

if($type eq "src")
{
  print '<srcset setid="sample_document_1" srclang="German">'."\n";
  wrap_each_file($ARGV[0]);
  print '</srcset>'."\n";
}
if($type eq "ref")
{
  for(my $i = 1; $i <= scalar @ARGV; $i++)
  {
    print '<refset setid="sample_document_1" srclang="German" trglang="English" refid="reference';
    print sprintf("%02d", $i);
    print '">'."\n";
    wrap_each_file($ARGV[$i-1]);
    print '</refset>'."\n";
  }
}
if($type eq "tst")
{
  print '<tstset setid="sample_document_1" srclang="German" trglang="English" sysid="Score">'."\n";
  wrap_each_file($ARGV[0]);
  print '</tstset>'."\n";
}

print '</mteval>'."\n";


sub wrap_each_file
{
  my ($file) = @_;

  print '<doc docid="sample_document_1" genre="temp">'."\n";

  open(F, $file);
  my $n = 1;
  while(my $line = <F>)
  {
    chomp($line);

    print "<seg id=\"".$n."\">".escape_xml($line)."<\/seg>\n";
    $n++;
  }

  close(F);

  print '</doc>'."\n";
}

sub escape_xml
{
  my ($text) = @_;

	$text =~ s/\&/\&amp;/g;   # escape escape
	$text =~ s/\|/\&#124;/g;  # factor separator
	$text =~ s/\</\&lt;/g;    # xml
	$text =~ s/\>/\&gt;/g;    # xml
	$text =~ s/\'/\&apos;/g;  # xml
	$text =~ s/\"/\&quot;/g;  # xml
	$text =~ s/\[/\&#91;/g;   # syntax non-terminal
	$text =~ s/\]/\&#93;/g;   # syntax non-terminal

  return $text;
}
