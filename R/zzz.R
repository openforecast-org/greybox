.onAttach <- function(libname, pkgname) {
    startUpMessage <- paste0("Package \"greybox\", v",packageVersion(pkgname)," loaded.");
    randomNumber <- trunc(runif(1,1,101));
    if(randomNumber<=4){
      # startUpMessage <- paste0(startUpMessage,"\n\033[38;2;00;70;20m");
      if(randomNumber==1){
        startUpMessage <- paste0(startUpMessage,"\nBy the way, have you already tried temporaldummy() function from greybox?");
      }
      else if(randomNumber==2){
        startUpMessage <- paste0(startUpMessage,"\nIf you want to know more about the greybox and forecasting, ",
                                 "you can visit my website: https://forecasting.svetunkov.ru/");
      }
      else if(randomNumber==3){
        startUpMessage <- paste0(startUpMessage,"\nAny thought or suggestions about the package? ",
                                 "Have you found a bug? File an issue on github: https://github.com/openforecast-org/greybox/issues");
      }
      else if(randomNumber==4){
        startUpMessage <- paste0(startUpMessage,"\nDid you know that you can use your own loss function in alm()? ",
                                 "This is regulated with 'loss' parameter. See documentation for examples.");
      }
    }
    startUpMessage <- paste0(startUpMessage,"\n");
    packageStartupMessage(startUpMessage);
}

.onUnload <- function (libpath) {
  library.dynam.unload("greybox", libpath);
}
