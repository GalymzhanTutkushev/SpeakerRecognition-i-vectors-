function  FrsOUT=NN_test_VAD(Y,Fs)

Y=Y-mean(Y);
Y=Y/max(abs(Y));
    p = floor(3*log(Fs));    
    preemph = [1 0.97];     
    DN=0.02*Fs;              
    NF=0.01*Fs;             
    Y = filter(1,preemph,Y);
             
   FrsIN = melcepst(Y,Fs,'dD',12,p,DN,NF);   % input
   FramesInBlock=0;
   NetFile = strcat('netMAD30.mat');
   load(NetFile,'net');
   FrsOUT = [];
   speech = net(FrsIN');

    ef=0;
    sizeX=0;
    ii=0;
      for s=1:size(FrsIN,1)-1

          if speech(s)>=0.7 && speech(s+1)>=0.7
             FramesInBlock=FramesInBlock+1;
          else
             if(FramesInBlock>6)  % 6*0.01=0.06c
               ii=ii+1;
               bf=ef+1;
               gran=size(FrsIN((s-FramesInBlock):(s-1),:));
               sizeX=gran(1)+sizeX;
               ef=bf+gran(1)-1;
               FrsOUT(bf:ef,:)=FrsIN((s-FramesInBlock):(s-1),:);
             end
            FramesInBlock=0; 
          end
      end
      FrsOUT=FrsOUT(1:sizeX,:);
      if size(FrsOUT,1)>1800  % 1800*0.01=18c
          
         FrsOUT=FrsOUT';
      else
           FrsOUT=[];
      end
end 
