
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from math import *


# In[2]:


#A beam is defined as [[px,py,pz],[qx,qy,qz]] where p is the direction and q is the position

class surface:
    """ Spherical surface  """
    #center=np.array([0,0,2.])
    #C=0.001    #1/Radius of curvature
    #n=1      #index of refraction
    #d=50     #diameter of the lens
       #position (Point where surface intersects z-Axis)
    #n0=1     #index of refraction before surface
    #k=0
    #DISZ = 
    def __init__(self,disz=1,curv=1E-20,n2=1,diameter=50,system=None):
        self.system=system  #pionting to the list of surfaces
        self.d=diameter
        self.n=n2                   #index of refraction after lens
        self.DISZ=disz              #distance to next lens
        self.set_Curv(curv)
        self.system=system
        self.k=0
    def pos(self):
        idx=self.system.index(self)
        if idx>0:
            prev=self.system[idx-1]
            if abs(prev.DISZ)==inf:
                return prev.pos()
            else:
                return prev.pos()+prev.DISZ
        else: 
            return 0
    def center(self):
        """ Return center of sphere"""
        return np.array([0,0,1/self.C+self.pos()])
    #def set_Power(self,power):
    #    """sets the refraction power. The radius is then (n2-n0)/power """
        #n2=self.n
        #n0=self.n0
        #self.C=1/((n2-n0+1E-30)/(power+1.E-40))
    def set_Curv(self,curv):
        """sets the refraction power. The radius is then (n2-n0)/power """
        #n0=self.n0
        self.C=(curv+1.E-40)
    def R(self):
        return 1/self.C
    def normal(self,points):   
        """return surface normals at points=[p1,p2,...,pn] where pi are points on surface """
        ez=np.array([[0,0,1]])
        v=((points-self.pos()*ez)*self.C-ez)
        return (v/np.linalg.norm(v,axis=1)[:,np.newaxis])#*np.sign(self.C)
    def plot(self,**kwargs):
        #R=self.R()
        C=self.C
        k=self.k
        ylim=1/max(abs(C)*1.05,2/self.d)
        y=np.linspace(-ylim,+ylim,100)
        #z=self.center()[2]-np.sign(c)*np.sqrt(R**2-y**2)
        z=C*y**2/(1+np.sqrt(1-(1+k)*C**2*y**2))+self.pos()
        plt.plot(z,y,**kwargs)
        
    #a beam is defined as q the position and P the direction
class beam_field:
   # Qa=np.array([[0,0,0],[0,0.1,0]])   #position of the beams [q1,q2,...,qn] where qi is the 3d pos of each beam
   # Pa=np.array([[0,0.1,1],[0,0.1,1]]) #direction of the beams [U1,U2,...,Un] 
    Q_p=None              #[Q1,Q2,...,QM] positions of the beam bundle where Qi=[q1,...,qn] is defined above 
    U=None #array([[[ux,uy,uz]]])  dim0=lensnr, dim1 =beamnr, dim2=coordinate
    n=np.array([1]) #indices of refraction
    def rectangular_grid(self,nx,ny,d,u=0):   #create an array of parallel beams d durchmesser 
        if nx == 1:
            dx=0
        else:
            dx=d
        dy=d
        if ny==1:
            dy=0
        xv, yv = np.mgrid[-dx:dx:nx*1j,-dy:dy:ny*1j]
        self.Q_p=np.array([np.array([xv.flatten(),yv.flatten(),np.zeros(nx*ny)]).T])        
        self.U=np.array([np.tile(np.array([0,u,1])/np.sqrt(u**2+1),(nx*ny,1))])
        self.normalize_U()
    def circular_pencil(self,n,d,pos,tip):
        """ creates a circular pencil beam field. 
        n circles of beams with 6*n beams in each circle start from tip=[px,py,pz]
        and have a diameter d at position pos
        """
        points=[[0,0,pos]]
        if n>0:
            r=d/2./n     
            b=[[sin(j)*r*i,cos(j)*r*i,pos] for i in range(1,n+1) for j in np.arange(0,2*pi,2*pi/6/i)]      
            points.extend(b)    
        self.Q_p=np.array([points])
        self.U=self.Q_p-tip
        self.normalize_U()
    def single_beam_from_Kingslake_Q(self,Qk,Uk):
        """ Create a single beam according to Q values defined
        Q=distance from  point where surface intersects with axis
        U angle against main axis
        """
        U=np.array([0,sin(Uk),cos(Uk)])
        Q=np.array([0,cos(Uk)*Qk,sin(Uk)*Qk])
        #transfer to starting point before lens
        Q=Q-U/np.cos(Uk)*np.sin(Uk)*Qk
        self.Q_p=Q[np.newaxis,np.newaxis,:]
        self.U=U[np.newaxis,np.newaxis,:]
    def normalize_U(self):
        self.U=self.U/np.linalg.norm(self.U,axis=2)[:,:,np.newaxis]
    def Kingslake_Qabs(self,surfaces):
        pos=np.array([s.pos() for s in surfaces])
        Q=self.Q_p[:-1]
        U=self.U[:-1]/np.linalg.norm(self.U[:-1],axis=2)[:,:,np.newaxis]
        A=np.array([0,0,1])*pos[:,np.newaxis,np.newaxis]
        Q=Q+np.sum((A-Q)*U,2)[:,:,np.newaxis]*U
        return(np.linalg.norm(Q-A,axis=2))
    def Kingslake_Q_abs(self,surfaces):
        pos=np.array([s.pos() for s in surfaces])
        Q=self.Q_p[1:]
        U=self.U[1:]/np.linalg.norm(self.U[1:],axis=2)[:,:,np.newaxis]
        A=np.array([0,0,1])*pos[:,np.newaxis,np.newaxis]
        Q=Q+np.sum((A-Q)*U,2)[:,:,np.newaxis]*U
        return(np.linalg.norm(Q-A,axis=2))
        
    def surface_points(self,surface):  #points where the beams hit the surface
        U=self.U[-1]
        U=U/np.linalg.norm(U,axis=1)[:,np.newaxis] 
        Q=self.Q_p[-1]
 #       QC=surface.center()-Q
 #       QCP=np.sum(QC*U,1)    #U normalized
#        QC2=np.sum(QC*QC,1) 
#        P2=np.sum(U*U,1)
        
#        a=(QCP-np.sign(surface.R())*np.sqrt(QCP*QCP+P2*(surface.R()**2-QC2)))/P2
        #variante 2
        C=surface.C
        a=surface.pos()                     #For P= surface point, C1=center of curv, A surface axis crossing
        ez=np.array([[0,0,1]])              #   which is then solved as d=r(X-sqrt(X**2-Y/R)) with x,Y helper functions
        X=U[:,2]-(np.sum(Q*U,1)-a*U[:,2])*C #X = helper coordinate 
        Y= np.sum((Q-a*ez)**2,axis=1)*C-2*(Q[:,2]-a)
        d=Y/(X+np.sqrt(X**2-C*Y))
        return U*d[:,np.newaxis]+Q
#        return (U.T*a).T+Q
    def refract(self,s,points):    
        #n0 index of refraction before surface
        #s surface at which the refraction takes place 
                       #for a normalized propagation vector P and a surface normal N and an outgoing P2
                            # the refraction laws can be written as n1*NxP=n2*NxP2
        #P=(self.P.T/np.linalg.norm(self.P,axis=1)).T #normalized propagation vectors                    
        P=self.U[-1] #normalized propagation vectors        
        #n1=np.linalg.norm(P,axis=1)[0]
        n1=self.n[-1]
        #print(n1)
        #print(np.linalg.norm(P,axis=1))
        P=P/np.linalg.norm(P,axis=1)[:,np.newaxis]
        n2=s.n
       
        N=s.normal(points)
        c=n1/n2*np.cross(N,P)                         #c= n1/n2 NxP,   equation c=NxP2 can be written as
        NC=np.cross(N,c)                              #P2=Nxc/N^2+b*N with b=sqrt(1-(NxC/N^2)^2) to normalize P2
        return -1*(NC+(np.sqrt(1-np.sum(NC*NC,1))*N.T).T)
        #print(self.P)
    def propagate(self,surfaces):
        self.Q_p=np.array([self.Q_p[0]])
        self.U=np.array([self.U[0]])
        self.n=np.array([self.n[0]])
        for s in surfaces:
            sp=np.array([self.surface_points(s)])
            
            self.Q_p=np.append(self.Q_p,sp,axis=0)
            newP=self.refract(s,sp[0])    
            #self.Q=sp[0]
            self.U=np.append(self.U,[newP],axis=0)
            self.n=np.append(self.n,[s.n])
    def plot(self,**kwargs):
        defaults={'color':'black','lw':0.1}
        kwargs={**defaults,**kwargs}    
        plt.plot(self.Q_p[:,:,2],self.Q_p[:,:,1],**kwargs)
    def calculate_intersections(self,P1,Q1,P2,Q2):
        """Intersection points between two beams"""
        x=(Q2[:,:,1]-Q1[:,:,1]-P2[:,:,1]/P2[:,:,2]*(Q2[:,:,2]-Q1[:,:,2]))/(P1[:,:,1]-P1[:,:,2]*P2[:,:,1]/P2[:,:,2])
        return Q1+x[:,:,np.newaxis]*P1,x
    def project_onto_plane(self,z): 
        """projects the last beam segment onto a plane"""
        U=self.U
        Q=self.Q_p
        #print(((z-Q[-2,:,[2]])/P[-2,:,[2]]).T)
        #print(P[-2])
        return ((z-Q[-2,:,[2]])/U[-2,:,[2]]).T*U[-2]+Q[-2]
       
    def circle_of_least_confusion(self,start):
        """Circle of least confusion also termed \Sigma_{LC}"""
        def f(x):
            pl=self.project_onto_plane(x)
            return max(pl[:,1])-min(pl[:,1])

 #       m=self.marginal_ray
        if hasattr(self, 'start'):
            start=self.start
        else:
#            start=(m.Q_p[-1,0,2]-m.Q_p[-2,0,2])/2
            start=start
        print(start)
        res=minimize(f,(start), method='Nelder-Mead')
        self.start=res.final_simplex[0][0,0]

        return res.final_simplex[0][0,0],res.final_simplex[1][0]


 
class paraxial:
    """Paraxial beam tracing"""
    y=np.array([1.])
    nu=np.array([0.])  # index of refraction n times inclination angle u=sin(u)  
    i=np.array([0.])
    def __init__(self,y,nu):
        """y = starting y value
           nu = index of refraction at start point times u(angle at start) 
        """
        self.y=np.array([y])
        self.nu=np.array([nu])
    def propagate(self,surfaces):
        y=self.y[[0]]
        nu=self.nu[[0]]
        self.i=self.i[[0]]
        nn=np.array([1])
        pos=0
        
        for s in surfaces:
            d=s.pos()-pos
            c=s.C
            n=nn[-1]
            n_=s.n  
            y_=y[-1]-d/n*nu[-1]
            y=np.append(y,[y_])
            #print(y)
            nu_=nu[-1]+y[-1]*(n_-n)*c
            self.i=np.append(self.i,[y[-1]*c-nu[-1]/n])
            nu=np.append(nu,[nu_])
            pos=s.pos()
            nn=np.append(nn,[n_])
        self.nu=nu
        self.y=y
        self.n_=nn         
    
        


from scipy.optimize import minimize
class lens_system:  
    """ Handle a whole system of lenses"""
    #surfaces=[]
    """list containing the surfaces of the lens system"""
    #entrance_pupil=11
    """diameter of entrance pupil"""
    marginal_ray=beam_field()
    """marginal_ray"""
    #n_init=1
    def __init__(self,pupil=20):
        """
        Parameters
            pupil:  diameter of entrance pupil
        """
        self.entrance_pupil=pupil
        self.surfaces=[]
        self.n_init=1
    def add_surface(self,s):
        """add a surface to the system. Surfacess must be added ordered by their z value."""
        self.surfaces.append(s)
        s.system=self.surfaces
    def plot(self,**kwargs):
        for s in self.surfaces:
            s.plot(**kwargs)
    def calculate_marginal(self):
        """calculate path of marginal ray"""
        self.marginal_ray=beam_field()
        m=self.marginal_ray
        m.U=np.array([[[0,0,1]]])
        m.Q_p=np.array([[[0,self.entrance_pupil,0]]])
        m.propagate(self.surfaces)
        
    def get_n(self):
        """return list with indices of refraction"""
        return np.append([self.n_init],[s.n for s in self.surfaces])
    def plot_caustic(self,**kwargs):
        defaults={'color':'black','lw':1}
        kwargs={**defaults,**kwargs} 
        f=beam_field()        
        f.rectangular_grid(1,20,self.entrance_pupil)
        f.propagate(self.surfaces)
        i,x=f.calculate_intersections(f.U[-2:-1,1:],f.Q_p[-2:-1,1:],f.U[-2:-1,:-1],f.Q_p[-2:-1,:-1])
        plt.plot(i[0,:,2],i[0,:,1],**kwargs)
    
    def circle_of_least_confusion(self):
        """Circle of least confusion also termed \Sigma_{LC}"""
        ff=beam_field()        
        ff.rectangular_grid(1,2000,self.entrance_pupil)
        ff.propagate(self.surfaces)
        def f(x):
            pl=ff.project_onto_plane(x)
            return max(pl[:,1])
            
 #       m=self.marginal_ray
        if hasattr(self, 'start'):
            start=self.start
        else:
#            start=(m.Q_p[-1,0,2]-m.Q_p[-2,0,2])/2
            start=(self.surfaces[-1].pos()-self.surfaces[-2].pos())/2
        #print(start)
        res=minimize(f,(start), method='Nelder-Mead')
        self.start=res.final_simplex[0][0,0]
        
        return res.final_simplex[0][0,0],res.final_simplex[1][0]
    def calculate_width_helper(self,d):
        m=self.marginal_ray

        n=beam_field()
        n.P_p=np.array([[[0,0,1]]])
        n.U=np.array([[[0,d,0]]])
        n.propagate(self.surfaces)
        i,x=m.calculate_intersections(m.U[-2:-1],m.Q_p[-2:-1],n.U[-2:-1],n.Q_p[-2:-1])
        
        return x[0,0],i
    def calculate_width(self):
        f=lambda x:-self.calculate_width_helper(x)[0]
        res=minimize(f, (-self.entrance_pupil*0.8), method='Nelder-Mead')
        d=res.final_simplex[0][0][0]
        i=self.calculate_width_helper(d)[1]
        #plt.plot(i[0,:,2],i[0,:,1],'o',color='black')
        #print(i[0,0,1])
        return i
    def Spherical_aberrations_surface_addup(self):
        """ the spherical aberrations are the distances between the zero crossings of a paraxial ray and
        the marginal ray. They are not calculated for each surface individually but as a total for a beam passing 
        the first n surfaces (experimental)"""
        bf=beam_field()
        bf.U=np.array([[[0,0,1],[0,0,1],[0,0,1]]])
        pp=self.entrance_pupil
        bf.Q_p=np.array([[[0,0,0],[0,pp/10.,0],[0,pp,0]]])    
        bf.propagate(self.surfaces)
        i,x=bf.calculate_intersections(bf.U[:,[0]],bf.Q_p[:,[0]],bf.U[:,[1]],bf.Q_p[:,[1]])
        
        i2,x2=bf.calculate_intersections(bf.U[:,[0]],bf.Q_p[:,[0]],bf.U[:,[2]],bf.Q_p[:,[2]])
        
        #print(i[:,:,2]-i2[:,:,2])
    def OSC(self,u):
        """angle tan: tangens of the paraxial beam angle u"""
        pr=paraxial(0,u)
        hnu=-u*self.entrance_pupil #n=1
        pr.propagate(self.surfaces)
        #print('hnu',hnu,1/hnu)
        #print('paraxial y ',pr.y[1:])
        #print('paraxial nu',pr.nu[:-1])
        #print('paraxial u ',pr.nu[:-1]/self.get_n()[:-1])
        #print('paraxial u ',pr.nu[:-1]/self.get_n()[:-1]/hnu/5.715023)
        #print('paraxial i ',pr.i[1:])
        marginal=beam_field()
        marginal.single_beam_from_Kingslake_Q(self.entrance_pupil,0)   #marginal beam
        marginal.propagate(self.surfaces)
        Q=marginal.Kingslake_Qabs(self.surfaces)[:,0]
        Q_=marginal.Kingslake_Q_abs(self.surfaces)[:,0]
        #print('marginal Q ',bb2.Kingslake_Qabs(ls.surfaces)[:,0])
        #print('marginal Q\'',bb2.Kingslake_Q_abs(ls.surfaces)[:,0])
        #print(Q-Q_)
        OSC1=(Q-Q_)*self.get_n()[:-1]*pr.i[1:]/hnu#*(u/sin(u))
        #print(self.get_n())
        #print('OSC contribution',OSC1)
        #print('sum',sum(OSC1))
        return OSC1
        #print('marginal sinU',bb2.U)
    def LA_contribution(self):
        """single plane contributions to spherical aberration"""
        pr=paraxial(self.entrance_pupil,0)
        #hnu=-u*self.entrance_pupil #n=1
        pr.propagate(self.surfaces)
        #print('hnu',hnu,1/hnu)
        #print('paraxial y ',pr.y[1:])
        #print('paraxial nu',pr.nu[:-1])
        #print('paraxial u ',pr.nu[:-1]/self.get_n()[:-1])
        #print('paraxial u ',pr.nu[:-1]/self.get_n()[:-1]/hnu/5.715023)
        #print('paraxial i ',pr.i[1:])
        ni=self.get_n()[:-1]*pr.i[1:]
        #print('ni',ni)
        marginal=beam_field()
        marginal.single_beam_from_Kingslake_Q(self.entrance_pupil,0)   #marginal beam
        marginal.propagate(self.surfaces)
        Q=marginal.Kingslake_Qabs(self.surfaces)[:,0]
        Q_=marginal.Kingslake_Q_abs(self.surfaces)[:,0]
        #print('marginal Q ',marginal.Kingslake_Qabs(ls.surfaces)[:,0])
        #print('marginal Q\'',marginal.Kingslake_Q_abs(ls.surfaces)[:,0])
        #print(Q-Q_)
        #print('paraxial nu\'',pr.nu[1:])
        #print('sin Uk\'',marginal.U)
        target_surface=len(self.surfaces)-1
        #print(marginal.U[3,0,1]*pr.nu[target_surface])
        nusinU=marginal.U[3,0,1]*pr.nu[target_surface] #n'u'sinU'_k all values at end focus
        LA=-(Q-Q_)*ni/nusinU
        #print('spherical LA contribution',LA)
        #print('sum',sum(LA))
        return LA
    def reverse(self):
        """reverses the system such that the first surface is the last"""
        sl=self.surfaces
        DISZ=sl[0].DISZ
        n=sl[0].n

        for s in sl[1:]:
            d=s.DISZ
            n0=s.n
            s.DISZ=DISZ
            s.n=n
            s.C*=-1
            DISZ=d
            n=n0
        
        sr=sl[::-1]
        a=[s for s in sr]
        self.surfaces.reverse()
        #print('done...')

